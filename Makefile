CC=nvcc
OUT=-o tsne
IN=tsne.cu

tsne: tsne.cu
	$(CC) $(OUT) $(IN)
clean: tsne
	rm tsne
