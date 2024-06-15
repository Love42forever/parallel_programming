
#include <iostream>
#include <vector>
#include<windows.h>
#include <stdio.h>
#include<typeinfo>
#include <stdlib.h>
#include<time.h>
#include<random>
#include<ctime>
#include <math.h>
#include <malloc.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>

#define N 4194304
#define THREAD_NUM 4
using namespace std;
int* data;
int* backup;
struct thread_param{
    int * arr;
    int low;
    int count;
    int tid;
};
// 双调合并
void init_data(){
    srand(static_cast<unsigned int>(time(0)));
    data= new int[N];
    backup= new int[N];
    for(int i=0;i<N;i++){
        data[i]=(rand()%10000);
        backup[i]=data[i];
    }
}
void reset_data(){
    for(int i=0;i<N;i++){
       data[i]=backup[i];
    }
}
void swap(int * arr,int i,int j){
    int temp = arr[i];
    arr[i]=arr[j];
    arr[j]=temp;
    return;
}
void avx_swap(int* arr,int i,int k,int dir){
    __m256i vec1 = _mm256_loadu_si256((__m256i*)&arr[i]);
    __m256i vec2 = _mm256_loadu_si256((__m256i*)&arr[i + k]);

    __m256i cmp_mask = _mm256_cmpgt_epi32(vec1, vec2);

    __m256i vec_min_data = _mm256_blendv_epi8(vec1, vec2, cmp_mask);
    __m256i vec_max_data = _mm256_blendv_epi8(vec2, vec1, cmp_mask);

    if(dir){
        _mm256_storeu_si256((__m256i*)&arr[i],vec_min_data);
        _mm256_storeu_si256((__m256i*)&arr[i+k],vec_max_data);
    }
    else{
        _mm256_storeu_si256((__m256i*)&arr[i],vec_max_data);
        _mm256_storeu_si256((__m256i*)&arr[i+k],vec_min_data);
    }
}
// 合并并排序两个双调序列
void bitonic_merge(int* arr, int low, int count, bool dir) {
    if (count > 1) {
        int k = count / 2;
        for (int i = low; i < low + k; i++) {
            if (dir == (arr[i] > arr[i + k]))
                swap(arr,i,i+k);
        }
        bitonic_merge(arr, low, k, dir);
        bitonic_merge(arr, low + k, k, dir);
    }
}

// 递归地创建双调序列
void bitonic_sort(int* arr, int low, int count, bool dir) {
    if (count > 1) {
        int k = count / 2;

        // 按递增顺序排列
        bitonic_sort(arr, low, k, true);

        // 按递减顺序排列
        bitonic_sort(arr, low + k, k, false);

        // 合并整个序列
        bitonic_merge(arr, low, count, dir);
    }
}
void avx_bitonic_merge(int* arr, int low, int count, bool dir) {
    if (count > 1) {
        int k = count / 2;
        if(k>=8){
            for(int i=low;i+8<=low+k;i+=8){
                avx_swap(arr,i,k,dir);
            }

            /*for (int i = low; i < low + k; i++) {
                if (dir == (arr[i] > arr[i + k]))
                    swap(arr,i,i+k);
            }*/
        }
        else{
            for (int i = low; i < low + k; i++) {
                if (dir == (arr[i] > arr[i + k]))
                    swap(arr,i,i+k);
            }
        }
        avx_bitonic_merge(arr, low, k, dir);
        avx_bitonic_merge(arr, low + k, k, dir);
    }
}
void avx_bitonic_sort(int* arr, int low, int count, bool dir) {
    if (count > 1) {
        int k = count / 2;

        // 按递增顺序排列
        avx_bitonic_sort(arr, low, k, true);

        // 按递减顺序排列
        avx_bitonic_sort(arr, low + k, k, false);

        // 合并整个序列
        avx_bitonic_merge(arr, low, count, dir);
    }
}
void* bitonic_sort_thread_func(void* param){
    thread_param * p = (thread_param*)param;
    int low = p->low;
    int count= p->count;
    int * arr = p->arr;
    bool dir = (p->tid+1)%2;
    bitonic_sort(arr,low,count,dir);
    pthread_exit(NULL);
    return NULL;
}
void bitonic_sort_thread(int* arr, int low, int count, bool dir){
        int thread_count=THREAD_NUM;
        pthread_t* handles=(pthread_t*)malloc(thread_count*sizeof(pthread_t));
        thread_param* params = (thread_param*)malloc(thread_count*sizeof(thread_param));
        int tid;
        int workload=N/THREAD_NUM;
        for(tid=0;tid<thread_count;tid++){
            params[tid].low=tid*workload;
            params[tid].count=workload;
            params[tid].tid=tid;
            params[tid].arr=arr;
            pthread_create(&handles[tid],NULL,bitonic_sort_thread_func,&params[tid]);
        }
        for(tid=0;tid<thread_count;tid++){
            pthread_join(handles[tid],NULL);
        }
        free(handles);
        free(params);
        bitonic_merge(arr,0,N/2,true);
        bitonic_merge(arr,N/2,N/2,false);
        bitonic_merge(arr,0,N,true);
}
void* avx_bitonic_sort_thread_func(void* param){
    thread_param * p = (thread_param*)param;
    int low = p->low;
    int count= p->count;
    int * arr = p->arr;
    bool dir = (p->tid+1)%2;
    avx_bitonic_sort(arr,low,count,dir);
    pthread_exit(NULL);
    return NULL;
}
void avx_bitonic_sort_thread(int* arr, int low, int count, bool dir){
        int thread_count=THREAD_NUM;
        pthread_t* handles=(pthread_t*)malloc(thread_count*sizeof(pthread_t));
        thread_param* params = (thread_param*)malloc(thread_count*sizeof(thread_param));
        int tid;
        int workload=N/THREAD_NUM;
        for(tid=0;tid<thread_count;tid++){
            params[tid].low=tid*workload;
            params[tid].count=workload;
            params[tid].tid=tid;
            params[tid].arr=arr;
            pthread_create(&handles[tid],NULL,avx_bitonic_sort_thread_func,&params[tid]);
        }
        for(tid=0;tid<thread_count;tid++){
            pthread_join(handles[tid],NULL);
        }
        free(handles);
        free(params);
        avx_bitonic_merge(arr,0,N/2,true);
        avx_bitonic_merge(arr,N/2,N/2,false);
        avx_bitonic_merge(arr,0,N,true);
}
int main() {

    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    init_data();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    bitonic_sort(data, 0, N, 1);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "Bitonic sort time:" << (tail - head) * 1000 / freq << "ms" << endl;
    cout << endl;
    reset_data();

    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    avx_bitonic_sort(data, 0, N, 1);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "AVX_Bitonic sort time:" << (tail - head) * 1000 / freq << "ms" << endl;
    cout << endl;
    reset_data();

    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    bitonic_sort_thread(data, 0, N, 1);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "pthread_Bitonic sort time:" << (tail - head) * 1000 / freq << "ms" << endl;
    cout << endl;
    reset_data();

    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    avx_bitonic_sort_thread(data, 0, N, 1);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "pthread_Bitonic sort time:" << (tail - head) * 1000 / freq << "ms" << endl;
    cout << endl;
    reset_data();
    /*int* data1= new int[N];
    vector<int> arr = {  9, 10, 12, 11, 1, 5, 3, 7, 2, 8,13, 15, 14, 16, 4, 6 };
    for(int i=0;i<N;i++){
        data1[i]=arr[i];
    }*/

    return 0;
}

