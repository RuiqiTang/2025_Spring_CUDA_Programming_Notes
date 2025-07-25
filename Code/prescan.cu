// Prefix Sum
'''
    Ref:https://zhuanlan.zhihu.com/p/661460705
'''
__global__ void prescan(float *g_odata,float *g_idata,int n){
    '''
        g_idata: input data, in global mem
        g_odata: output data, in global mem
    '''
    extern __shared__ float temp[] // 在调用时分配

    int thid=threadIdx.x;
    int offset=1;

    // load input to shared mem
    temp[2*thid]=g_idata[2*thid];
    temp[2*thid+1]=g_idata[2*thid+1];

    // build sum in place up the tree
    for (int d=n>>1;d>0;d>=1){
        __syncthreads()

        if (thid<d){
            int ai=offset*(2*thid+1)-1;
            int bi=offset*(2*thid+2)-1;

            temp[bi]+=temp[ai];
        }
        offset*=2
    }

    // clear the last element
    if(thid==0){temp[n-1]=0;}

    // traverse down tree & build scan
    for (int d=1;d<n;d*2){

    }
}