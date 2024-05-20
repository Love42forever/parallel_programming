#include<iostream>
#include<windows.h>
#include <stdio.h>
#include<typeinfo>
#include <stdlib.h>
#include<time.h>
#include<random>
#include<ctime>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
using namespace std;
#define N 2000
float ** unaligned_matrix;
float ** aligned_matrix;
void init_data(){
    mt19937 rng(static_cast<unsigned int>(time(0)));
    uniform_real_distribution<float> dist(0.0f, 10.0f);
    unaligned_matrix = new float*[N];
    for(int i=0;i<N;i++){
        unaligned_matrix[i]= new float[N];
        for(int j=0;j<N;j++){
            unaligned_matrix[i][j]=dist(rng);
        }
    }
}
void clear_data(){
    for(int i=0;i<N;i++){
        delete[] unaligned_matrix[i];
    }
    delete unaligned_matrix;
}

void clear_data_aligned(){
    for(int i=0;i<N;i++){
        delete[] aligned_matrix[i];
    }
    delete aligned_matrix;
}
void init_data_aligned(int alignment){
    mt19937 rng(static_cast<unsigned int>(time(0)));
    uniform_real_distribution<float> dist(0.0f, 10.0f);
    aligned_matrix = new float*[N];
    for (size_t i = 0; i < N; ++i) {
        aligned_matrix[i] = static_cast<float*>(_aligned_malloc(sizeof(float)*N,alignment));
        if (aligned_matrix[i] == nullptr) {
            std::cerr << "aligned_alloc failed" << std::endl;
            for (size_t j = 0; j < i; ++j) {
                free(aligned_matrix[j]);
            }
            delete[] aligned_matrix;
            return;
        }
                for(int j=0;j<N;j++){
            aligned_matrix[i][j]=dist(rng);
        }
    }
}
void LU(float** matrix){
    for(int k=0;k<N;k++){
        for(int j=k+1;j<N;j++){
            matrix[k][j]=matrix[k][j]/matrix[k][k];
        }
        matrix[k][k]=1.0;
        for(int i=k+1;i<N;i++){
            for(int j=k+1;j<N;j++){
                matrix[i][j]=matrix[i][j]-matrix[k][j]*matrix[i][k];
            }
            matrix[i][k]=0.0;
        }
    }
}
void LU_sse(float** matrix){
    for(int k=0;k<N;k++){
        __m128 t1=_mm_set1_ps(matrix[k][k]);
        int j=0;
        for(j=k+1;j+4<=N;j+=4){
            __m128 t2 = _mm_loadu_ps(&matrix[k][j]);
            t2 = _mm_div_ps(t2,t1);
            _mm_storeu_ps(&matrix[k][j],t2);
        }
        while(j<N){
            matrix[k][j]=matrix[k][j]/matrix[k][k];
            j++;
        }
        matrix[k][k]=1.0;
        for(int i=k+1;i<N;i++){
            __m128 ik=_mm_set1_ps(matrix[i][k]);
            for(j=k+1;j+4<=N;j+=4){
                __m128 kj=_mm_loadu_ps(&matrix[k][j]);
                __m128 temp = _mm_mul_ps(ik,kj);
                __m128 ij = _mm_loadu_ps(&matrix[i][j]);
                ij = _mm_sub_ps(ij,temp);
                _mm_storeu_ps(&matrix[i][j],ij);
            }
            while(j<N){
                matrix[i][j]=matrix[i][j]-matrix[k][j]*matrix[i][k];
                j++;
            }
            matrix[i][k]=0.0;
        }
    }
}
void LU_avx(float** matrix){
    for(int k=0;k<N;k++){
        __m256 t1=_mm256_set1_ps(matrix[k][k]);
        int j=0;
        for(j=k+1;j+8<=N;j+=8){
            __m256 t2 = _mm256_loadu_ps(&matrix[k][j]);
            t2 = _mm256_div_ps(t2,t1);
            _mm256_storeu_ps(&matrix[k][j],t2);
        }
        while(j<N){
            matrix[k][j]=matrix[k][j]/matrix[k][k];
            j++;
        }
        matrix[k][k]=1.0;
        for(int i=k+1;i<N;i++){
            __m256 ik=_mm256_set1_ps(matrix[i][k]);
            for(j=k+1;j+8<=N;j+=8){
                __m256 kj=_mm256_loadu_ps(&matrix[k][j]);
                __m256 temp = _mm256_mul_ps(ik,kj);
                __m256 ij = _mm256_loadu_ps(&matrix[i][j]);
                ij = _mm256_sub_ps(ij,temp);
                _mm256_storeu_ps(&matrix[i][j],ij);
            }
            while(j<N){
                matrix[i][j]=matrix[i][j]-matrix[k][j]*matrix[i][k];
                j++;
            }
            matrix[i][k]=0.0;
        }
    }
}
void LU_sse_aligned(float** matrix){
    for(int k=0;k<N;k++){
        __m128 t1=_mm_set1_ps(matrix[k][k]);
        int j=k+1;
        int offset=reinterpret_cast<uintptr_t>(&matrix[k][j])%16;
        while((offset!=0)&&j<N){
            matrix[k][j]=matrix[k][j]/matrix[k][k];
            j++;
            offset=reinterpret_cast<uintptr_t>(&matrix[k][j])%16;
        }
        for(;j+4<=N;j+=4){
            __m128 t2 = _mm_load_ps(&matrix[k][j]);
            t2 = _mm_div_ps(t2,t1);
            _mm_store_ps(&matrix[k][j],t2);
        }
        while(j<N){
            matrix[k][j]=matrix[k][j]/matrix[k][k];
            j++;
        }
        matrix[k][k]=1.0;
        for(int i=k+1;i<N;i++){
            __m128 ik=_mm_set1_ps(matrix[i][k]);
            int j=k+1;
            int offset=reinterpret_cast<uintptr_t>(&matrix[k][j])%16;
            while((offset!=0)&&j<N){
                matrix[i][j]=matrix[i][j]-matrix[k][j]*matrix[i][k];
                j++;
                offset=reinterpret_cast<uintptr_t>(&matrix[k][j])%16;
            }
            for(;j+4<=N;j+=4){
                __m128 kj=_mm_load_ps(&matrix[k][j]);
                __m128 temp = _mm_mul_ps(ik,kj);
                __m128 ij = _mm_load_ps(&matrix[i][j]);
                ij = _mm_sub_ps(ij,temp);
                _mm_store_ps(&matrix[i][j],ij);
            }
            while(j<N){
                matrix[i][j]=matrix[i][j]-matrix[k][j]*matrix[i][k];
                j++;
            }
            matrix[i][k]=0.0;
        }
    }
}
void LU_avx_aligned(float** matrix){
    for(int k=0;k<N;k++){
        __m256 t1=_mm256_set1_ps(matrix[k][k]);
        int j=k+1;
        int offset=reinterpret_cast<uintptr_t>(&matrix[k][j])%32;
        while((offset!=0)&&j<N){
            matrix[k][j]=matrix[k][j]/matrix[k][k];
            j++;
            offset=reinterpret_cast<uintptr_t>(&matrix[k][j])%32;
        }
        for(;j+8<=N;j+=8){
            __m256 t2 = _mm256_load_ps(&matrix[k][j]);
            t2 = _mm256_div_ps(t2,t1);
            _mm256_store_ps(&matrix[k][j],t2);
        }
        while(j<N){
            matrix[k][j]=matrix[k][j]/matrix[k][k];
            j++;
        }
        matrix[k][k]=1.0;
        for(int i=k+1;i<N;i++){
            __m256 ik=_mm256_set1_ps(matrix[i][k]);
            int j=k+1;
            int offset=reinterpret_cast<uintptr_t>(&matrix[k][j])%32;
            while((offset!=0)&&j<N){
                matrix[i][j]=matrix[i][j]-matrix[k][j]*matrix[i][k];
                j++;
                offset=reinterpret_cast<uintptr_t>(&matrix[k][j])%32;
            }
            for(;j+8<=N;j+=8){
                __m256 kj=_mm256_load_ps(&matrix[k][j]);
                __m256 temp = _mm256_mul_ps(ik,kj);
                __m256 ij = _mm256_load_ps(&matrix[i][j]);
                ij = _mm256_sub_ps(ij,temp);
                _mm256_store_ps(&matrix[i][j],ij);
            }
            while(j<N){
                matrix[i][j]=matrix[i][j]-matrix[k][j]*matrix[i][k];
                j++;
            }
            matrix[i][k]=0.0;
        }
    }
}
int main() {

    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    init_data();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    LU(unaligned_matrix);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "LU time:" << (tail - head) * 1000 / freq << "ms" << endl;
    clear_data();
//
//
    init_data();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    LU_sse(unaligned_matrix);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    //print(unalign);
    cout<<"LU_sse time:"<<(tail-head)*1000/freq<<"ms"<<endl;
    clear_data();

    init_data_aligned(16);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    LU_sse_aligned(aligned_matrix);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout <<"LU_sse_aligned time:" << (tail - head) * 1000 / freq << "ms" << endl;

    init_data();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    LU_avx(unaligned_matrix);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    //print(unalign);
    cout<<"LU_avx time:"<<(tail-head)*1000/freq<<"ms"<<endl;
    clear_data();

    init_data_aligned(32);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    LU_avx_aligned(aligned_matrix);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout <<"LU_avx_aligned time:" << (tail - head) * 1000 / freq << "ms" << endl;

    return 0;
}
