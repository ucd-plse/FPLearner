// create-space Clang plugin
//
#include <iostream>
#include <fstream>
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "../utilities/json.hpp"


using namespace clang;
using namespace std;
using namespace llvm;

#define endline "\n"
#define VERBOSE 1
#define PRINT_DEBUG_MESSAGE(s) if (VERBOSE > 0) {errs() << s << endline; }

string funcName;
//std::string basefilename;
nlohmann::json event;
nlohmann::json searchspace;
nlohmann::json includespace;
int varCount = 0;
string GLOBAL = "globalVar";
string LOCAL = "localVar";
string PARM = "parmVar";
string CALL = "call";
string OUTPUT_PATH = "";
string OUTPUT_NAME = "config.json";
string INPUT_FILE = "";
string INCLUDE = "include.json";

// for string delimiter
vector<string> split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

// Used by std::find_if
struct MatchPathSeparator
{
    bool operator()(char ch) const {
        return ch == '/';
    }
};

string basename(std::string path) {
    return std::string( std::find_if(path.rbegin(), path.rend(), MatchPathSeparator()).base(), path.end());
}

struct FloatingPointTypeInfo {
    bool isFloatingPoint : 1;
    unsigned int isVector : 3;
    bool isPointer : 1;
    bool isArray : 1;
    const clang::Type* typeObj;
};

FloatingPointTypeInfo DissectFloatingPointType(const clang::Type* typeObj, bool builtIn) {
    if (typeObj == NULL) {
        FloatingPointTypeInfo info;
        info.isFloatingPoint = false;
        info.isVector = 0;
        info.typeObj = NULL;
        info.isArray = false;
        info.isPointer = false;
        return info;
    }
    FloatingPointTypeInfo info;
    info.isArray = false;
    info.isPointer = false;
    if (const clang::ArrayType* arr = dyn_cast<clang::ArrayType>(typeObj)) {
//        PRINT_DEBUG_MESSAGE("\t\tis array type");
        info = DissectFloatingPointType(arr->getElementType().getTypePtrOrNull(), false);
        info.isArray = true;
        return info;
    }
    else if (const clang::PointerType* ptr = dyn_cast<clang::PointerType>(typeObj)) {
//        PRINT_DEBUG_MESSAGE("\t\tis pointer type");
        info = DissectFloatingPointType(ptr->getPointeeType().getTypePtrOrNull(), false);
        info.isPointer = true;
        return info;
    }
    else if (const clang::PointerType* ptr = dyn_cast<clang::PointerType>(typeObj->getCanonicalTypeInternal())) {
//        PRINT_DEBUG_MESSAGE("\t\tis pointer type");
        info = DissectFloatingPointType(ptr->getPointeeType().getTypePtrOrNull(), false);
        info.isPointer = false;
        return info;
    }

//    PRINT_DEBUG_MESSAGE("\t\tinnermost type " << typeObj->getCanonicalTypeInternal().getAsString());

    if (const clang::BuiltinType* bltin = dyn_cast<clang::BuiltinType>(typeObj)) {
//        PRINT_DEBUG_MESSAGE("\t\tis builtin type, floating point: " << bltin->isFloatingPoint());
        info.isFloatingPoint = bltin->isFloatingPoint();
        info.isVector = 0;
        info.typeObj = typeObj;
        return info;
    }
    else if (typeObj->isStructureType()) {
//        PRINT_DEBUG_MESSAGE("\t\tis struct type");
        // TODO: with floating point built-in vectors and __half
        info.isFloatingPoint = false;
        info.isVector = 0;
        std::string typeStr = typeObj->getCanonicalTypeInternal().getAsString();
        if (typeStr == "struct __half") {
            info.isFloatingPoint = true;
            info.isVector = 0;
        }
        else {
            size_t pos = typeStr.find("struct float");
            if (pos != std::string::npos) {
                info.isFloatingPoint = true;
                info.isVector = typeStr[pos + strlen("struct float")] - '1';
            }
            pos = typeStr.find("struct double");
            if (pos != std::string::npos) {
                info.isFloatingPoint = true;
                info.isVector = typeStr[pos + strlen("struct double")] - '1';
            }
        }
        info.typeObj = typeObj;
        return info;
    }
    else {
//        PRINT_DEBUG_MESSAGE("\t\tis another type");
        info.isFloatingPoint = false;
        info.isVector = 0;
        info.typeObj = typeObj;
        info.isArray = false;
        info.isPointer = false;
        return info;
    }
}

namespace {
class TraverseGlobalVisitor : public clang::RecursiveASTVisitor<TraverseGlobalVisitor> {
public:
  explicit TraverseGlobalVisitor(ASTContext *Context) : Context(Context) {};
  bool VisitVarDecl(clang::VarDecl *val) {
    if (val->hasGlobalStorage()) {
        const clang::Type* typeObj = val->getType().getTypePtrOrNull();
        if (typeObj) {
            FloatingPointTypeInfo info = DissectFloatingPointType(typeObj, true);

            bool FPStrMatch = false;
            size_t pos1 = val->getType().getAsString().find("double");
            size_t pos2 = val->getType().getAsString().find("float");
            if (pos1 != std::string::npos || pos2 != std::string::npos) {
                FPStrMatch = true;
            }

            if (info.isFloatingPoint || FPStrMatch) {
                string valueName = val->getNameAsString();
                string typeName = val->getType().getAsString();

                SourceManager& SrcMgr = Context->getSourceManager();
                const FileEntry* Entry = SrcMgr.getFileEntryForID(SrcMgr.getFileID(val->getLocation()));
                string FileName = basename(Entry->getName().str());
                // if this global variable is to consider or not
                if (FileName == INPUT_FILE && includespace.contains(FileName)) {
                    if (std::find(includespace[FileName]["global"].begin(), includespace[FileName]["global"].end(), valueName) != includespace[FileName]["global"].end())
                    {
                        // get line number
                        string linenum = "";
                        auto loc = val->getLocation();
                        string source_loc = loc.printToString(Context->getSourceManager());
                        string delimiter = ":";
                        vector<string> v = split (source_loc, delimiter);
                        int count = 0;
                        for (auto i : v) {
                                    count = count + 1;
                                    if (count == 2) {
                                        linenum = i;
                                    }
                        } 

                        varCount ++;
                        event[GLOBAL+to_string(varCount)]["name"] = valueName;
                        event[GLOBAL+to_string(varCount)]["type"] = typeName;
                        event[GLOBAL+to_string(varCount)]["location"] = GLOBAL;
                        event[GLOBAL+to_string(varCount)]["lines"] = {linenum};
                        event[GLOBAL+to_string(varCount)]["file"] = FileName;
                        string next_type = typeName;
                        next_type.replace(0, 6, "float");
                        searchspace[GLOBAL+to_string(varCount)]["name"] = valueName;
                        searchspace[GLOBAL+to_string(varCount)]["type"] = {next_type, typeName};
                        searchspace[GLOBAL+to_string(varCount)]["location"] = GLOBAL;
                        searchspace[GLOBAL+to_string(varCount)]["lines"] = {linenum};
                        searchspace[GLOBAL+to_string(varCount)]["file"] = FileName;
                    }
                }
            }
        }
    }
    return true;
  }
private:
  clang::ASTContext *Context;
};

class TraverseVarsVisitor : public clang::RecursiveASTVisitor<TraverseVarsVisitor> {
public:
  explicit TraverseVarsVisitor(ASTContext *Context) : Context(Context) {};

  bool VisitVarDecl(clang::VarDecl *val) {
        SourceManager& SrcMgr = Context->getSourceManager();
        const FileEntry* Entry = SrcMgr.getFileEntryForID(SrcMgr.getFileID(val->getLocation()));
        string FileName = basename(Entry->getName().str());

        const clang::Type* typeObj = val->getType().getTypePtrOrNull();
        if (typeObj) {
            FloatingPointTypeInfo info = DissectFloatingPointType(typeObj, true);

            bool FPStrMatch = false;
            size_t pos1 = val->getType().getAsString().find("double");
            size_t pos2 = val->getType().getAsString().find("float");
            if (pos1 != std::string::npos || pos2 != std::string::npos) {
                FPStrMatch = true;
            }

            if (info.isFloatingPoint || FPStrMatch) {
                string valueName = val->getNameAsString();

                if (std::find(includespace[FileName]["function"][funcName].begin(), includespace[FileName]["function"][funcName].end(), valueName) == includespace[FileName]["function"][funcName].end())
                {
                    string typeName = val->getType().getAsString();

                    // get line number
                    string linenum = "";
                    auto loc = val->getBeginLoc();
                    string source_loc = loc.printToString(Context->getSourceManager());
                    string delimiter = ":";
                    vector<string> v = split (source_loc, delimiter);
                    int count = 0;
                    for (auto i : v) {
                        count = count + 1;
                        if (count == 2) {
                            linenum = i;
                        }
                    } 

                    varCount ++;
                    event[LOCAL+to_string(varCount)]["function"] = funcName;
                    event[LOCAL+to_string(varCount)]["name"] = valueName;
                    event[LOCAL+to_string(varCount)]["type"] = typeName;
                    event[LOCAL+to_string(varCount)]["location"] = LOCAL;
                    event[LOCAL+to_string(varCount)]["file"] = FileName;
                    event[LOCAL+to_string(varCount)]["lines"] = {linenum};

                    string next_type = typeName;
                    if (next_type.substr(0, 5) == "const") {
                        next_type.replace(6, 6, "float");
                    } else {
                        next_type.replace(0, 6, "float");
                    }
                    searchspace[LOCAL+to_string(varCount)]["function"] = funcName;
                    searchspace[LOCAL+to_string(varCount)]["name"] = valueName;
                    searchspace[LOCAL+to_string(varCount)]["type"] = {next_type, typeName};
                    searchspace[LOCAL+to_string(varCount)]["location"] = LOCAL;
                    searchspace[LOCAL+to_string(varCount)]["file"] = FileName;
                    searchspace[LOCAL+to_string(varCount)]["lines"] = {linenum};

                }
            }
        }
        return true;
  }
private:
  clang::ASTContext *Context;
};

class FuncStmtVisitor : public RecursiveASTVisitor<FuncStmtVisitor> {
public:
    explicit FuncStmtVisitor(ASTContext *Context) : Context(Context) {};

    bool VisitStmt(clang::Stmt *st) {
        if (const clang::DeclRefExpr* declSt = dyn_cast<clang::DeclRefExpr>(st)) {
            if (const clang::FunctionDecl* libcall = dyn_cast<clang::FunctionDecl>(declSt->getDecl())){
                if (libcall->getNameAsString() == "sqrt" 
                || libcall->getNameAsString() == "log" 
                || libcall->getNameAsString() == "sin" 
                || libcall->getNameAsString() == "cos"
                // || libcall->getNameAsString() == "fabs" 
                || libcall->getNameAsString() == "exp" ) {
                    string valueName = libcall->getNameAsString();
                    string switchName = libcall->getNameAsString();
                    varCount ++;
                    event[CALL+to_string(varCount)]["function"] = funcName;
                    event[CALL+to_string(varCount)]["name"] = valueName;
                    event[CALL+to_string(varCount)]["switch"] = switchName;

                    auto loc = declSt->getLocation();
                    event[CALL+to_string(varCount)]["location"] = loc.printToString(Context->getSourceManager());

                    // get line number
                    string linenum = "";

                    string source_loc = loc.printToString(Context->getSourceManager());
                    string delimiter = ":";
                    vector<string> v = split (source_loc, delimiter);
                    int count = 0;
                    for (auto i : v) {
                                count = count + 1;
                                if (count == 2) {
                                    linenum = i;
                                }
                    } 
                    event[CALL+to_string(varCount)]["lines"] = {linenum};

                    SourceManager& SrcMgr = Context->getSourceManager();
                    const FileEntry* Entry = SrcMgr.getFileEntryForID(SrcMgr.getFileID(declSt->getLocation()));
                    string FileName = basename(Entry->getName().str());
                    event[CALL+to_string(varCount)]["file"] = FileName;
                    event[CALL+to_string(varCount)]["type"] = {"double", "double"};

                    string next_switch = switchName+"f";
                    searchspace[CALL+to_string(varCount)]["function"] = funcName;
                    searchspace[CALL+to_string(varCount)]["name"] = valueName;
                    searchspace[CALL+to_string(varCount)]["switch"] = {next_switch, switchName};
                    searchspace[CALL+to_string(varCount)]["location"] = loc.printToString(Context->getSourceManager());
                    searchspace[CALL+to_string(varCount)]["lines"] = {linenum};
                    searchspace[CALL+to_string(varCount)]["file"] = FileName;
                    searchspace[CALL+to_string(varCount)]["type"] = nlohmann::json::array({{"float", "float"}, {"double", "double"}});
                }
            }
            else if (const clang::VarDecl* val = dyn_cast<clang::VarDecl>(declSt->getDecl())){
                    const clang::Type* typeObj = val->getType().getTypePtrOrNull();
                    if (typeObj) {
                        FloatingPointTypeInfo info = DissectFloatingPointType(typeObj, true);

                        bool FPStrMatch = false;
                        size_t pos1 = val->getType().getAsString().find("double");
                        size_t pos2 = val->getType().getAsString().find("float");
                        if (pos1 != std::string::npos || pos2 != std::string::npos) {
                            FPStrMatch = true;
                        }

                        if (info.isFloatingPoint && FPStrMatch) {
                            string valueName = val->getNameAsString();
                            string typeName = val->getType().getAsString();
                            string linenum = "";

                            auto loc = st->getBeginLoc();
                            string source_loc = loc.printToString(Context->getSourceManager());
                            string delimiter = ":";
                            vector<string> v = split (source_loc, delimiter);
                            int count = 0;
                            for (auto i : v) {
                                count = count + 1;
                                if (count == 2) {
                                    linenum = i;
                                }

                            } 


                            // todo:
                            for (nlohmann::json::iterator it = event.begin(); it != event.end(); ++it) {
                                if (event[it.key()]["function"] == funcName && event[it.key()]["name"] == valueName){
                                    if (event[it.key()]["lines"].empty()) {
                                        event[it.key()]["lines"] = {linenum};
                                        searchspace[it.key()]["lines"] = {linenum};
                                    }  else {
                                        if (std::find(event[it.key()]["lines"].begin(), event[it.key()]["lines"].end(), linenum) == event[it.key()]["lines"].end()) {
                                            event[it.key()]["lines"].push_back(linenum);
                                            searchspace[it.key()]["lines"].push_back(linenum);
                                        }
                                    }
                                    
                                }
                            }


                        }
                        
                    }
                    
            }
        }
        return true;
    }
private:
    ASTContext *Context;
};

class TraverseFuncVisitor : public clang::RecursiveASTVisitor<TraverseFuncVisitor> {
public:
  explicit TraverseFuncVisitor(ASTContext *Context) : VarsVisitor(Context), st_visitor(Context), Context(Context) {};

  bool VisitFunctionDecl(FunctionDecl* func) {
    if (!func->doesThisDeclarationHaveABody())
        return true;

    funcName = func->getNameInfo().getName().getAsString();
    // errs() << "Function: " << funcName << "\n";

    // get file name
    SourceManager& SrcMgr = Context->getSourceManager();
    const FileEntry* Entry = SrcMgr.getFileEntryForID(SrcMgr.getFileID(func->getLocation()));
    string FileName = basename(Entry->getName().str());

    if (FileName != INPUT_FILE ) {
        return true;
    }

    // if this function is to consider or not
    if (includespace.contains(FileName)) {
        if (includespace[FileName]["function"].contains(funcName)) {
            for (ParmVarDecl* param : func->parameters()) {
                    const clang::Type* typeObj = param->getType().getTypePtrOrNull();
                    if (typeObj) {
                        FloatingPointTypeInfo info = DissectFloatingPointType(typeObj, true);
                        if (info.isFloatingPoint) {
                            string valueName = param->getNameAsString();

                            if (std::find(includespace[FileName]["function"][funcName].begin(), includespace[FileName]["function"][funcName].end(), valueName) == includespace[FileName]["function"][funcName].end())
                            {
                                string typeName = param->getOriginalType().getAsString();
                                
                                // get line number
                                string linenum = "";
                                auto loc = param->getLocation();
                                string source_loc = loc.printToString(Context->getSourceManager());
                                string delimiter = ":";
                                vector<string> v = split (source_loc, delimiter);
                                int count = 0;
                                for (auto i : v) {
                                            count = count + 1;
                                            if (count == 2) {
                                                linenum = i;
                                            }
                                } 
                                
                                varCount ++;
                                event[LOCAL+to_string(varCount)]["function"] = funcName;
                                event[LOCAL+to_string(varCount)]["name"] = valueName;
                                event[LOCAL+to_string(varCount)]["type"] = typeName;
                                event[LOCAL+to_string(varCount)]["location"] = PARM;
                                event[LOCAL+to_string(varCount)]["lines"] = {linenum};
                                event[LOCAL+to_string(varCount)]["file"] = FileName;

                                string next_type = typeName;
                                // next_type.replace(0, 6, "float");
                                // todo
                                if (next_type.substr(0, 5) == "const") {
                                    next_type.replace(6, 6, "float");
                                } else {
                                    next_type.replace(0, 6, "float");
                                }

                                searchspace[LOCAL+to_string(varCount)]["function"] = funcName;
                                searchspace[LOCAL+to_string(varCount)]["name"] = valueName;
                                searchspace[LOCAL+to_string(varCount)]["type"] = {next_type, typeName};
                                searchspace[LOCAL+to_string(varCount)]["location"] = PARM;
                                searchspace[LOCAL+to_string(varCount)]["lines"] = {linenum};
                                searchspace[LOCAL+to_string(varCount)]["file"] = FileName;
                            }

                        }
                    }
            }

            VarsVisitor.TraverseStmt(func->getBody());
            st_visitor.TraverseStmt(func->getBody());
        }
    }

    return true;
  }
private:
  TraverseVarsVisitor VarsVisitor;
  FuncStmtVisitor st_visitor;
  clang::ASTContext *Context;
};

class TraverseFuncVarsConsumer : public clang::ASTConsumer {
public:
  explicit TraverseFuncVarsConsumer(clang::ASTContext *Context, clang::DiagnosticsEngine *Diagnostics)
    : FuncVisitor(Context), GlobVisitor(Context) {}

  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    std::ifstream ifs(INCLUDE);
    includespace = nlohmann::json::parse(ifs);

    GlobVisitor.TraverseDecl(Context.getTranslationUnitDecl());
    FuncVisitor.TraverseDecl(Context.getTranslationUnitDecl());
    // Create an output file to write the updated code
    string filename = OUTPUT_PATH + OUTPUT_NAME;
    std::error_code OutErrorInfo;
    std::error_code ok;
    llvm::raw_fd_ostream outFile(llvm::StringRef(filename),
                OutErrorInfo, llvm::sys::fs::F_None);
    if (OutErrorInfo == ok) {
                outFile << std::string(event.dump(4));
                PRINT_DEBUG_MESSAGE("Output file created - " << filename);
        } else {
                PRINT_DEBUG_MESSAGE("Could not create file - " << filename);
        }

    string sp_filename = OUTPUT_PATH + "search_" + OUTPUT_NAME;
    std::error_code sp_OutErrorInfo;
    std::error_code sp_ok;
    llvm::raw_fd_ostream sp_outFile(llvm::StringRef(sp_filename),
                sp_OutErrorInfo, llvm::sys::fs::F_None);
    if (sp_OutErrorInfo == sp_ok) {
                sp_outFile << std::string(searchspace.dump(4));
                PRINT_DEBUG_MESSAGE("Output file created - " << sp_filename);
        } else {
                PRINT_DEBUG_MESSAGE("Could not create file - " << sp_filename);
        }


  }
private:
  TraverseFuncVisitor FuncVisitor;
  TraverseGlobalVisitor GlobVisitor;
};

class TraverseFuncVarsAction : public clang::PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef) {
    return std::make_unique<TraverseFuncVarsConsumer>(&CI.getASTContext(), &CI.getDiagnostics());
  }

  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string>& args) {
    // To be written...
    llvm::errs() << "Plugin arg size = " << args.size() << "\n";
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      if (args[i] == "-output-path") {
        if (i+1 < e)
            OUTPUT_PATH = args[i+1];
        else
            PRINT_DEBUG_MESSAGE("Missing output path! Could not generate output file.");
        llvm::errs() << "Output path = " << OUTPUT_PATH << "\n";
      }
      if (args[i] == "-output-name") {
        if (i+1 < e)
            OUTPUT_NAME = args[i+1];
        else
            PRINT_DEBUG_MESSAGE("Missing output name! Could not generate output file.");
        llvm::errs() << "Output file name = " << OUTPUT_NAME << "\n";
      }
      if (args[i] == "-input-file") {
        if (i+1 < e)
            INPUT_FILE = args[i+1];
        else
            PRINT_DEBUG_MESSAGE("Missing input file name! Could not generate output file.");
        llvm::errs() << "Input file name = " << INPUT_FILE << "\n";
      }
    }
    return true;
  }
};

}

static clang::FrontendPluginRegistry::Add<TraverseFuncVarsAction>
X("create-space", "find all variables in each function");
