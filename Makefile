CC = gcc
CFLAGS = -Wall -Wextra -I./include -O3 -MMD -MP
AR = ar
ARFLAGS = rcs

SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
DEPS = $(OBJS:.o=.d)

STATIC_LIB = $(LIB_DIR)/libtenCor.a

all: dirs $(STATIC_LIB)

dirs:
	@mkdir -p $(OBJ_DIR) $(LIB_DIR)

$(STATIC_LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)

-include $(DEPS)

.PHONY: all dirs clean