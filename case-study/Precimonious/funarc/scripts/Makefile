CC = gcc
CFLAGS = -lm
SOURCES = funarc.c
EXECUTABLE = funarc

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f $(EXECUTABLE) *.txt *.out
