// trans-type Clang plugin
//

#include "TransformType.h"
#include <iostream>
#include <fstream>
#include <string>
#include "../utilities/json.hpp"

using namespace clang;
using namespace std;
using namespace llvm;

nlohmann::json jf;
string funcName;
string parm_Name = "";
string type_Name = "";
string new_TypeName = "";
string OUTPUT_PATH = "";
string INPUT_CONFIG = "config_temp.json";
int libcall_flag = 0;


size_t countSubString(const string& str, const string& sub) {
   size_t ret = 0;
   size_t loc = str.find(sub);
   while (loc != string::npos) {
      ++ret;
      loc = str.find(sub, loc+1);
   }
   return ret;
}

void PrintSourceRange(SourceRange range, ASTContext* astContext) {
    PRINT_DEBUG_MESSAGE("\toffset: " << astContext->getSourceManager().getFileOffset(range.getBegin()) << " " <<
    astContext->getSourceManager().getFileOffset(range.getEnd()));
}

void PrintStatement(string prefix, const Stmt* st, ASTContext* astContext) {
    std::string statementText;
    raw_string_ostream wrap(statementText);
    st->printPretty(wrap, NULL, PrintingPolicy(astContext->getLangOpts()));
    PRINT_DEBUG_MESSAGE(prefix << st->getStmtClassName() << ", " << statementText);
    PrintSourceRange(st->getSourceRange(), astContext);
}

string getFileName(ASTContext *Context, VarDecl *val) {
    SourceManager& SrcMgr = Context->getSourceManager();
    const FileEntry* Entry = SrcMgr.getFileEntryForID(SrcMgr.getFileID(val->getLocation()));
    string FileName = basename(Entry->getName().str());
    return FileName;
}

string getFileName(ASTContext *Context, DeclRefExpr *val) {
    SourceManager& SrcMgr = Context->getSourceManager();
    const FileEntry* Entry = SrcMgr.getFileEntryForID(SrcMgr.getFileID(val->getLocation()));
    string FileName = basename(Entry->getName().str());
    return FileName;
}

string getNewType(string valueName, string typeName, string funcName, string fileName, string location, string source_loc) {
    for (nlohmann::json::iterator it = jf.begin(); it != jf.end(); ++it) {
      if (location == "call") {
          if (valueName == jf[it.key()]["name"]
          && funcName == jf[it.key()]["function"]
          && fileName == jf[it.key()]["file"]
          && source_loc == jf[it.key()]["location"]) {
            if (typeName == jf[it.key()]["switch"]) {
                return "";
            }
            else {
                return jf[it.key()]["switch"];
            }
          }
      }
      else if (location != "globalVar") {
          if (valueName == jf[it.key()]["name"]
          && funcName == jf[it.key()]["function"]
          && fileName == jf[it.key()]["file"]
          && location == jf[it.key()]["location"]) {
            if (typeName == jf[it.key()]["type"]) {
                return "";
            }
            else {
                return jf[it.key()]["type"];
            }
          }
      }
      else {
          if (valueName == jf[it.key()]["name"]
          && fileName == jf[it.key()]["file"]
          && location == jf[it.key()]["location"]) {
            if (typeName == jf[it.key()]["type"]) {
                return "";
            }
            else {
                return jf[it.key()]["type"];
            }
          }
      }

    }
    return "";
}

string basename(std::string path) {
    return std::string( std::find_if(path.rbegin(), path.rend(), MatchPathSeparator()).base(), path.end());
}

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

void transformType(string location, VarDecl *val, ASTContext *astContext) {
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
                string typeName = "";
                if (location == "parmVar")
                    typeName = dyn_cast<clang::ParmVarDecl>(val)->getOriginalType().getAsString();
                else {
                    typeName = val->getType().getAsString();
                    string fileName = getFileName(astContext, val);
                    string newTypeName = getNewType(valueName, typeName, funcName, fileName, location, "");
                    if (newTypeName != "") {
                        rewriter.ReplaceText(val->getTypeSourceInfo()->getTypeLoc().getBeginLoc(), 6, "float");
//                        if (typeName.substr(0, 5) == "const") {
//                            rewriter.ReplaceText(val->getTypeSourceInfo()->getTypeLoc().getBeginLoc(), typeName.length(), newTypeName);
//                        } else {
//                            rewriter.ReplaceText(val->getTypeSourceInfo()->getTypeLoc().getBeginLoc(), 6, "float");
//                        }

//                        if (info.isPointer) {
//        //                  newTypeName += "float * ";
//                            rewriter.ReplaceText(val->getTypeSourceInfo()->getTypeLoc().getBeginLoc(), 6, "float");
//                        } else {
//        //                  newTypeName += "float ";
////                            int replaceLen = (typeName.length() > 6) ? 6 : typeName.length();
//                            rewriter.ReplaceText(val->getTypeSourceInfo()->getTypeLoc().getBeginLoc(), 6, newTypeName.substr(0,6));
//
//                        }
                    }
                }

            }
        }
}

void transformSwitch(clang::DeclRefExpr* declSt, clang::FunctionDecl* libcall, ASTContext *astContext) {
    string valueName = libcall->getNameAsString();
    string switchName = libcall->getNameAsString();
    string fileName = getFileName(astContext, declSt);
    auto loc = declSt->getLocation();
    string source_loc = loc.printToString(astContext->getSourceManager());
    string newTypeName = getNewType(valueName, switchName, funcName, fileName, "call", source_loc);
    if (newTypeName != "") {
        SourceLocation ST = loc.getLocWithOffset(valueName.length());
        string tobeinsert = "f";
        rewriter.InsertText(ST, tobeinsert, true, true);
    }

}

bool GlobVisitor::VisitVarDecl(VarDecl *val){
    if (val->hasGlobalStorage()) {
        transformType("globalVar", val, astContext);
    }
    return true;
}

bool VarsVisitor::VisitVarDecl(VarDecl *val){
    transformType("localVar", val, astContext);
    return true;
}

bool FuncStmtVisitor::VisitStmt(Stmt *st) {
//    PrintStatement("Statement: ", st, astContext);
    if ( clang::DeclRefExpr* declSt = dyn_cast<clang::DeclRefExpr>(st)) {
        if (libcall_flag == 0) {
            if ( clang::VarDecl* var = dyn_cast<clang::VarDecl>(declSt->getDecl())){
                if (var->getNameAsString() == parm_Name) {
                    rewriter.InsertText(declSt->getLocation(), parm_Name, true, true);
                }
            }
        } else {
            if ( clang::FunctionDecl* libcall = dyn_cast<clang::FunctionDecl>(declSt->getDecl())){
                if (libcall->getNameAsString() == "sqrt" 
                || libcall->getNameAsString() == "log" 
                || libcall->getNameAsString() == "sin" 
                || libcall->getNameAsString() == "cos"
                // || libcall->getNameAsString() == "fabs" 
                || libcall->getNameAsString() == "exp" ){
                    transformSwitch(declSt, libcall, astContext);
                }
            }
        }
    }
    return true;
}

bool FuncVisitor::VisitFunctionDecl(FunctionDecl* func) {
    if (!func->doesThisDeclarationHaveABody())
        return true;

    funcName = func->getNameInfo().getName().getAsString();

    libcall_flag = 0;
    for (ParmVarDecl* param : func->parameters()) {
        const clang::Type* typeObj = param->getType().getTypePtrOrNull();
        if (typeObj) {
            FloatingPointTypeInfo info = DissectFloatingPointType(typeObj, true);
            if (info.isFloatingPoint) {
                string valueName = param->getNameAsString();
                string typeName = dyn_cast<clang::ParmVarDecl>(param)->getOriginalType().getAsString();
                string fileName = getFileName(astContext, param);
                string newTypeName = getNewType(valueName, typeName, funcName, fileName, "parmVar", "");
//                if (1) {
                if (newTypeName != "") {
                    parm_Name = valueName;
                    type_Name = typeName;
                    new_TypeName = newTypeName;

                    SourceLocation ST = func->getBody()->getSourceRange().getBegin().getLocWithOffset(1);
                    string tobeinsert = "";
                    if (type_Name.back() == ']' || type_Name.back() == '*') {
                        int count1 = countSubString(type_Name, "]");
                        int count2 = countSubString(type_Name, "*");
                        if (count1 == 1 || count2 == 1) {
                            tobeinsert = "\n  float * " + parm_Name + parm_Name + " = (float *)" + parm_Name + ";\n";
                        } else if (count1 == 2 || count2 == 2) {
                            tobeinsert = "\n  float ** " + parm_Name + parm_Name + " = (float **)" + parm_Name + ";\n";
                        } else if (count1 == 3 || count2 == 3) {
                            tobeinsert = "\n  float *** " + parm_Name + parm_Name + " = (float ***)" + parm_Name + ";\n";
                        } else if (count1 == 4 || count2 == 4) {
                            tobeinsert = "\n  float **** " + parm_Name + parm_Name + " = (float ****)" + parm_Name + ";\n";
                        }

                    } else {
                        tobeinsert = "\n  float " + parm_Name + parm_Name + " = " + parm_Name + ";\n";
                    }
                    rewriter.InsertText(ST, tobeinsert, true, true);


                    st_visitor->TraverseStmt(func->getBody());


                }

            }
        }
    }
    libcall_flag = 1;
    st_visitor->TraverseStmt(func->getBody());
    vars_visitor->TraverseStmt(func->getBody());

    return true;

}

void TransTypeConsumer::HandleTranslationUnit(ASTContext &Context){
    FileID id = rewriter.getSourceMgr().getMainFileID();
    string basefilename = basename(rewriter.getSourceMgr().getFilename(rewriter.getSourceMgr().getLocForStartOfFile(id)).str());
    string filename = OUTPUT_PATH + basefilename;
//    PRINT_DEBUG_MESSAGE(filename);
    std::ifstream ifs(INPUT_CONFIG);
    jf = nlohmann::json::parse(ifs);

    glob_visitor->TraverseDecl(Context.getTranslationUnitDecl());
    func_visitor->TraverseDecl(Context.getTranslationUnitDecl());

    // Create an output file to write the updated code
    std::error_code OutErrorInfo;
    std::error_code ok;
    const RewriteBuffer *RewriteBuf = rewriter.getRewriteBufferFor(id);
    if (RewriteBuf) {
        llvm::raw_fd_ostream outFile(llvm::StringRef(filename),
                OutErrorInfo, llvm::sys::fs::F_None);
        if (OutErrorInfo == ok) {
                outFile << std::string(RewriteBuf->begin(), RewriteBuf->end());
                PRINT_DEBUG_MESSAGE("Output file created - " << filename);
        } else {
                PRINT_DEBUG_MESSAGE("Could not create file - " << filename);
        }
    }else {
        PRINT_DEBUG_MESSAGE("No file created!");
    }
}

unique_ptr<ASTConsumer> TransTypeAction::CreateASTConsumer(CompilerInstance &CI, StringRef file) {
    return make_unique<TransTypeConsumer>(&CI);
}

bool TransTypeAction::ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string>& args) {
    // To be written...
    llvm::errs() << "Plugin arg size = " << args.size() << "\n";
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      if (args[i] == "-output-path") {
        if (i+1 < e)
            OUTPUT_PATH = args[i+1]; // ./temp-NPB3.3-SER-C/CG/
        else
            PRINT_DEBUG_MESSAGE("Missing output path! Could not generate output file.");
        llvm::errs() << "Output path = " << OUTPUT_PATH << "\n";
      }
      if (args[i] == "-input-config") {
        if (i+1 < e)
            INPUT_CONFIG = args[i+1]; // config_temp.json
        else
            PRINT_DEBUG_MESSAGE("Missing input config! Could not generate output file.");
        llvm::errs() << "Input config  = " << INPUT_CONFIG << "\n";
      }
    }
    return true;
  }

static clang::FrontendPluginRegistry::Add<TransTypeAction>
X("trans-type", "transform all floating point variables' type from double to float");