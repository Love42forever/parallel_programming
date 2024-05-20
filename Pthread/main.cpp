#include<iostream>
#include <stdio.h>
#include<typeinfo>
#include <stdlib.h>
#include<time.h>
#include<random>
#include<ctime>
#include<semaphore.h>
#include<pthread.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include<windows.h>
using namespace std;
#define N 2000
#define THREAD_NUM 4
float ** data;
struct thread_param{
    int k;
    int tid;
};
void init_data(){
    mt19937 rng(static_cast<unsigned int>(time(0)));
    uniform_real_distribution<float> dist(0.0f, 10.0f);
    data = new float*[N];
    for(int i=0;i<N;i++){
        data[i]= new float[N];
        for(int j=0;j<N;j++){
            data[i][j]=dist(rng);
        }
    }
}
void clear_data(){
    for(int i=0;i<N;i++){
        delete[] data[i];
    }
    delete data;
}
//void init_data_aligned(int alignment){
//    mt19937 rng(static_cast<unsigned int>(time(0)));
//    uniform_real_distribution<float> dist(0.0f, 10.0f);
//    aligned_matrix = new float*[N];
//    for (size_t i = 0; i < N; ++i) {
//        aligned_matrix[i] = static_cast<float*>(_aligned_malloc(sizeof(float)*N,alignment));
//        if (aligned_matrix[i] == nullptr) {
//            std::cerr << "aligned_alloc failed" << std::endl;
//            for (size_t j = 0; j < i; ++j) {
//                free(aligned_matrix[j]);
//            }
//            delete[] aligned_matrix;
//            return;
//        }
//                for(int j=0;j<N;j++){
//            aligned_matrix[i][j]=dist(rng);
//        }
//    }
//}
void LU( ){
    for(int k=0;k<N;k++){
        for(int j=k+1;j<N;j++){
            data[k][j]=data[k][j]/data[k][k];
        }
        data[k][k]=1.0;
        for(int i=k+1;i<N;i++){
            for(int j=k+1;j<N;j++){
                data[i][j]=data[i][j]-data[k][j]*data[i][k];
            }
            data[i][k]=0.0;
        }
    }
}
void* LU_thread_func(void* param){
    thread_param * p = (thread_param*)param;
    int k = p->k;
    int tid= p->tid;
    int i = k+1+tid;
    for(int j=k+1;j<N;j++){
        data[i][j]=data[i][j]-data[i][k]*data[k][j];
    }
    data[i][k]=0.0;
    pthread_exit(NULL);
    return NULL;
}
void LU_pthread(){
    for(int k=0;k<N;k++){
        for(int j=k+1;j<N;j++){
            data[k][j]=data[k][j]/data[k][k];
        }
        data[k][k]=1.0;

        int thread_count=N-1-k;
        pthread_t* handles=(pthread_t*)malloc(thread_count*sizeof(pthread_t));
        thread_param* params = (thread_param*)malloc(thread_count*sizeof(thread_param));
        int tid;
        for(tid=0;tid<thread_count;tid++){
            params[tid].tid=tid;
            params[tid].k=k;
            pthread_create(&handles[tid],NULL,LU_thread_func,&params[tid]);
        }
        for(tid=0;tid<thread_count;tid++){
            pthread_join(handles[tid],NULL);
        }
        free(handles);
        free(params);

    }
}
void* LU_thread_func_fixed(void* param){
    thread_param * p = (thread_param*)param;
    int k = p->k;
    int tid= p->tid;
    int i = k+1+tid;
    for(;i<N;i+=THREAD_NUM){
        for(int j=k+1;j<N;j++){
            data[i][j]=data[i][j]-data[i][k]*data[k][j];
        }
        data[i][k]=0.0;
    }
    pthread_exit(NULL);
    return NULL;
}
void LU_pthread_fixed(){
    for(int k=0;k<N;k++){
        for(int j=k+1;j<N;j++){
            data[k][j]=data[k][j]/data[k][k];
        }
        data[k][k]=1.0;

        int thread_count=THREAD_NUM;
        pthread_t* handles=(pthread_t*)malloc(thread_count*sizeof(pthread_t));
        thread_param* params = (thread_param*)malloc(thread_count*sizeof(thread_param));
        int tid;
        for(tid=0;tid<thread_count;tid++){
            params[tid].tid=tid;
            params[tid].k=k;
            pthread_create(&handles[tid],NULL,LU_thread_func_fixed,&params[tid]);
        }
        for(tid=0;tid<thread_count;tid++){
            pthread_join(handles[tid],NULL);
        }
        free(handles);
        free(params);

    }
}
void* LU_thread_func_successive(void* param){
    thread_param * p = (thread_param*)param;
    int k = p->k;
    int tid= p->tid;
    int row_num=(N-k)/THREAD_NUM;
    int i = k+1+tid*row_num;
    int num=0;
    for(;i<N;i++){
        for(int j=k+1;j<N;j++){
            data[i][j]=data[i][j]-data[i][k]*data[k][j];
        }
        data[i][k]=0.0;
        num++;
        if(num==row_num){
            break;
        }
    }
    pthread_exit(NULL);
    return NULL;
}
void LU_pthread_fixed_successive(){
    for(int k=0;k<N;k++){
        for(int j=k+1;j<N;j++){
            data[k][j]=data[k][j]/data[k][k];
        }
        data[k][k]=1.0;

        int thread_count=THREAD_NUM;
        pthread_t* handles=(pthread_t*)malloc(thread_count*sizeof(pthread_t));
        thread_param* params = (thread_param*)malloc(thread_count*sizeof(thread_param));
        int tid;
        for(tid=0;tid<thread_count;tid++){
            params[tid].tid=tid;
            params[tid].k=k;
            pthread_create(&handles[tid],NULL,LU_thread_func_successive,&params[tid]);
        }
        for(tid=0;tid<thread_count;tid++){
            pthread_join(handles[tid],NULL);
        }
        free(handles);
        free(params);

    }
}
void* avx_thread_func_fixed(void* param){
        thread_param * p = (thread_param*)param;
        int k = p->k;
        int tid= p->tid;
        int i = k+1+tid;
        for(;i<N;i+=THREAD_NUM){
            __m256 ik = _mm256_set1_ps(data[i][k]);
            int j;
            for(j=k+1;j+8<=N;j+=8){
                __m256 kj=_mm256_loadu_ps(&data[k][j]);
                __m256 temp = _mm256_mul_ps(ik,kj);
                __m256 ij = _mm256_loadu_ps(&data[i][j]);
                ij = _mm256_sub_ps(ij,temp);
                _mm256_storeu_ps(&data[i][j],ij);
            }
            data[i][k]=0.0;
            while(j<N){
                data[i][j]=data[i][j]-data[k][j]*data[i][k];
                j++;
            }
        }
        pthread_exit(NULL);
        return NULL;
}
void avx_pthread_fixed(){
    for(int k=0;k<N;k++){
        __m256 kk = _mm256_set1_ps(data[k][k]);
        int j;
        for(j=k+1;j+8<=N;j+=8){
            __m256 kj = _mm256_loadu_ps(&data[k][j]);
            kj=_mm256_div_ps(kj,kk);
            _mm256_storeu_ps(&data[k][j],kj);
        }
        while(j<N){
            data[k][j]=data[k][j]/data[k][k];
            j++;
        }
        data[k][k]=1.0;

        int thread_count=THREAD_NUM;
        pthread_t* handles=(pthread_t*)malloc(thread_count*sizeof(pthread_t));
        thread_param* params = (thread_param*)malloc(thread_count*sizeof(thread_param));
        int tid;
        for(tid=0;tid<thread_count;tid++){
            params[tid].tid=tid;
            params[tid].k=k;
            pthread_create(&handles[tid],NULL,avx_thread_func_fixed,&params[tid]);
        }
        for(tid=0;tid<thread_count;tid++){
            pthread_join(handles[tid],NULL);
        }
        free(handles);
        free(params);

    }
}
void* avx_thread_func_fixed_successive(void* param){
        thread_param * p = (thread_param*)param;
        int k = p->k;
        int tid= p->tid;
        int row_num=(N-k)/THREAD_NUM;
        int i = k+1+tid*row_num;
        int num=0;
        for(;i<N;i++){
            __m256 ik = _mm256_set1_ps(data[i][k]);
            int j;
            for(j=k+1;j+8<=N;j+=8){
                __m256 kj=_mm256_loadu_ps(&data[k][j]);
                __m256 temp = _mm256_mul_ps(ik,kj);
                __m256 ij = _mm256_loadu_ps(&data[i][j]);
                ij = _mm256_sub_ps(ij,temp);
                _mm256_storeu_ps(&data[i][j],ij);
            }
            while(j<N){
                data[i][j]=data[i][j]-data[k][j]*data[i][k];
                j++;
            }
            num++;
            if(num==row_num)
                break;
        }
        pthread_exit(NULL);
        return NULL;
}
void avx_pthread_fixed_successive(){
    for(int k=0;k<N;k++){
        __m256 kk = _mm256_set1_ps(data[k][k]);
        int j;
        for(j=k+1;j+8<=N;j+=8){
            __m256 kj = _mm256_loadu_ps(&data[k][j]);
            kj=_mm256_div_ps(kj,kk);
            _mm256_storeu_ps(&data[k][j],kj);
        }
        while(j<N){
            data[k][j]=data[k][j]/data[k][k];
            j++;
        }
        data[k][k]=1.0;

        int thread_count=THREAD_NUM;
        pthread_t* handles=(pthread_t*)malloc(thread_count*sizeof(pthread_t));
        thread_param* params = (thread_param*)malloc(thread_count*sizeof(thread_param));
        int tid;
        for(tid=0;tid<thread_count;tid++){
            params[tid].tid=tid;
            params[tid].k=k;
            pthread_create(&handles[tid],NULL,avx_thread_func_fixed_successive,&params[tid]);
        }
        for(tid=0;tid<thread_count;tid++){
            pthread_join(handles[tid],NULL);
        }
        free(handles);
        free(params);

    }
}
void call_func(void(*func)(),char* funcname){
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    init_data();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    func();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout <<funcname<< " time:" << (tail - head) * 1000 / freq << "ms" << endl;
    clear_data();
}

int main(){
    call_func(LU,"LU");
    //call_func(LU_pthread,"LU pthread");
    call_func(LU_pthread_fixed,"LU pthread fixed");
    call_func(LU_pthread_fixed_successive,"LU pthread fixed successive");
    call_func(avx_pthread_fixed,"avx pthread fixed");
    call_func(avx_pthread_fixed_successive,"avx pthread fixed successive");
    return 0;
}
