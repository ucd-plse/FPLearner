include Makefile.config

PROG		= $(PLUGIN).so
SRC_FILES	= $(PLUGIN).cpp

O_FILES		= $(SRC_FILES:%.cpp=%.o)

all: $(PROG)

$(PROG): $(O_FILES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(PROG) $(SHARED_LIB_OPT) $(O_FILES)

clean:
	$(RM) *.o $(PROG) *.so
