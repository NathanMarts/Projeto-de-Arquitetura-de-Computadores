#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

#define TAM 1000

float m1[TAM*TAM], m2[TAM*TAM], r[TAM*TAM];

int main(int argc, char *argv[]){
	int aux = TAM*TAM, aux2 = TAM, temp = 0;
	
	for (int i = 0, w = 0; i < TAM; i++){
    	for (int j = 0; j < TAM; ++j, w++){
    		m1[w] = j+1;
    	}
	}

	for (int i = 0, j = 1; i < aux; ++i){
		if(i > aux2-1){
			j++;
			aux2 += TAM;
		}
		m2[i] = j;
	}

	clock_t Ticks[2];
    Ticks[0] = time(NULL);
    __m256 x1;
	__m256 x2;
	__m256 x3;
	#pragma omp parallel for private(temp,x1,x2,x3,x) shared(m1,m2,r)
    for(int k = 0, x = 0; k < aux; k+=TAM, x++){
	    for(int h = 0; h < aux; h+=TAM){
			for(int i = 0; i < TAM; i+=8){
				x1 = _mm256_load_ps((float*)m1+i+k);
				x2 = _mm256_load_ps((float*)m2+i+h);
				x3 = _mm256_mul_ps(x1, x2);
				for (int w = 0; w < 8; ++w)
					temp += x3[w];
			}
			r[x] = temp;
			temp = 0;
	    }
	}
	
	Ticks[1] = time(NULL);
    double Tempo = difftime(Ticks[1], Ticks[0]);
    printf("Tempo gasto: %g Seg.\n", Tempo);
    
	return 0;
}