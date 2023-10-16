#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang;
using namespace std;
using namespace llvm;

#define endline "\n"
#define VERBOSE 1
#define PRINT_DEBUG_MESSAGE(s) if (VERBOSE > 0) {errs() << s << endline; }

// Used by std::find_if
struct MatchPathSeparator
{
    bool operator()(char ch) const {
        return ch == '/';
    }
};

struct FloatingPointTypeInfo {
    bool isFloatingPoint : 1;
    unsigned int isVector : 3;
    bool isPointer : 1;
    bool isArray : 1;
    const clang::Type* typeObj;
};

// Function to get the base name of the file provided by path
string basename(std::string path);
FloatingPointTypeInfo DissectFloatingPointType(const clang::Type* typeObj, bool builtIn);

Rewriter rewriter;

class VarsVisitor : public RecursiveASTVisitor<VarsVisitor> {
private:
    ASTContext *astContext; // used for getting additional AST info

public:
    explicit VarsVisitor(CompilerInstance *CI)
        :   astContext(&(CI->getASTContext())) // initialize private members
    {
        rewriter.setSourceMgr(astContext->getSourceManager(),
            astContext->getLangOpts());
    }

    virtual bool VisitVarDecl(VarDecl *val);
};

class FuncStmtVisitor : public RecursiveASTVisitor<FuncStmtVisitor> {
private:
    ASTContext *astContext; // used for getting additional AST info

public:
    explicit FuncStmtVisitor(CompilerInstance *CI)
        :   astContext(&(CI->getASTContext())) // initialize private members
    {
        rewriter.setSourceMgr(astContext->getSourceManager(),
            astContext->getLangOpts());
    }

    virtual bool VisitStmt(Stmt *st);
};

class FuncVisitor : public RecursiveASTVisitor<FuncVisitor> {
private:
    VarsVisitor *vars_visitor;
    FuncStmtVisitor *st_visitor;
    ASTContext *astContext; // used for getting additional AST info

public:
    explicit FuncVisitor(CompilerInstance *CI)
        :  vars_visitor(new VarsVisitor(CI)), st_visitor(new FuncStmtVisitor(CI)), astContext(&(CI->getASTContext())) {}

    virtual bool VisitFunctionDecl(FunctionDecl* func);
};

class GlobVisitor : public RecursiveASTVisitor<GlobVisitor> {
private:
    ASTContext *astContext; // used for getting additional AST info

public:
    explicit GlobVisitor(CompilerInstance *CI)
        :  astContext(&(CI->getASTContext())) {}

    virtual bool VisitVarDecl(VarDecl* val);
};

class TransTypeConsumer : public ASTConsumer {
private:
    FuncVisitor *func_visitor; // doesn't have to be private
    GlobVisitor *glob_visitor; // doesn't have to be private

public:
    explicit TransTypeConsumer(CompilerInstance *CI)
        : func_visitor(new FuncVisitor(CI)), glob_visitor(new GlobVisitor(CI)) {}

    virtual void HandleTranslationUnit(ASTContext &Context);
};

class TransTypeAction : public PluginASTAction {
protected:
    unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file);
    bool ParseArgs(const CompilerInstance &CI, const vector<string> &args);
};