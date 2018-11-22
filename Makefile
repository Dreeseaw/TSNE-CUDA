CC=nvcc
CFLAGS=
NAME=tsne

.PHONY: all
all:
	$(CC) $(CFLAGS) -o $(NAME) $(NAME).cu

.PHONY: clean
clean:
	rm -f $(name)
