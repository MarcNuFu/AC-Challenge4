#include <stdio.h>
#include <stdlib.h>

#include <xmmintrin.h>

// Define here as constant for easy change
#define REAL double
unsigned int seed;

// Function to generate pseudo-random numbers
int myRandom() {
  seed = (214013*seed+2531011);
  return (seed>>3);
}



void printCheck ( REAL V[], int N )
{
  int x;

  REAL S=0;
  for (x=0; x<=N+1; x++)
    S = S + V[x];

  printf("\nCheckSum = %1.10e\n", S);

  //for (x=0; x<8; x++)
  //  printf("%1.10e\n", V[x]);
  for (x=0; x<=8; x++)
    printf("%1.10e\n", V[x*N/8]);
}

void SimulationStep ( REAL * __restrict In, REAL L, int N, REAL bot, int diff, REAL Factor)
{
  REAL temp = 1/L - 2.0f;
  REAL InAnterior = bot;
  REAL InActual = In[0];
  REAL InFuturo;
  REAL LFactor = L*Factor;
  
 _mm_prefetch((const char*) &In[8], _MM_HINT_NTA);
 _mm_prefetch((const char*) &In[16], _MM_HINT_NTA);
 _mm_prefetch((const char*) &In[24], _MM_HINT_NTA);
 _mm_prefetch((const char*) &In[32], _MM_HINT_NTA); 
 
  for (int x=0; x<N-diff; x+=16)
  {
    for(int j=0; j<16; j++)
    {
      InFuturo = In[x+j+1];
      In[x+j] = LFactor*(InActual*temp + InFuturo + InAnterior);
      InAnterior = InActual;
      InActual = InFuturo;
    }
    _mm_prefetch((const char*) &In[x+40], _MM_HINT_NTA);
    _mm_prefetch((const char*) &In[x+48], _MM_HINT_NTA);        
  }
  
  for (int x=N-diff; x<N; x++)
  {
      InFuturo = In[x+1];
      In[x] = LFactor*(InActual*temp + InFuturo + InAnterior);
      InAnterior = InActual;
      InActual = InFuturo;
  }
}


void SimulationStep2 ( REAL * __restrict In, REAL L, int N, REAL bot, int diff, REAL Factor1, REAL Factor2)
{
  REAL temp = 1/L - 2.0f;
  REAL InAnterior = bot;
  REAL InAnterior2 = bot;
  REAL InActual = In[0];
  REAL InActual2;
  REAL InFuturo, InFuturo2;
  
  REAL LFactor1 = L*Factor1;
  REAL LFactor2 = L*Factor2;
  
   _mm_prefetch((const char*) &In[8], _MM_HINT_NTA);
   _mm_prefetch((const char*) &In[16], _MM_HINT_NTA);
   _mm_prefetch((const char*) &In[24], _MM_HINT_NTA);
   _mm_prefetch((const char*) &In[32], _MM_HINT_NTA); 
 
 
   for(int j=0; j<16; j++)
   {
      InFuturo = In[j+1];
      In[j] = LFactor1*(InActual*temp + InFuturo + InAnterior);
      InAnterior = InActual;
      InActual = InFuturo;
   }
  _mm_prefetch((const char*) &In[40], _MM_HINT_NTA);
  _mm_prefetch((const char*) &In[48], _MM_HINT_NTA);
  
  
  
  InActual2 = In[0];
  for (int x=16; x<N-diff; x+=16)
  {
    for(int j=0; j<16; j++)
    {
      InFuturo = In[x+j+1];
      In[x+j] = LFactor1*(InActual*temp + InFuturo + InAnterior);
      InAnterior = InActual;
      InActual = InFuturo;
    }
    
    for(int j=0; j<16; j++)
    {
      InFuturo2 = In[x+j-15];
      In[x+j-16] = LFactor2*(InActual2*temp + InFuturo2 + InAnterior2);
      InAnterior2 = InActual2;
      InActual2 = InFuturo2;
    }
    
    _mm_prefetch((const char*) &In[x+40], _MM_HINT_NTA);
    _mm_prefetch((const char*) &In[x+48], _MM_HINT_NTA);        
  }
  
  
  
  for (int x=N-diff; x<N; x++)
  {
      InFuturo = In[x+1];
      In[x] = LFactor1*(InActual*temp + InFuturo + InAnterior);
      InAnterior = InActual;
      InActual = InFuturo;
  }
  
  
  for(int j=N-diff-16; j<N-diff; j++)
  {
    InFuturo2 = In[j+1];
    In[j] = LFactor2*(InActual2*temp + InFuturo2 + InAnterior2);
    InAnterior2 = InActual2;
    InActual2 = InFuturo2;
  }
  
  for (int x=N-diff; x<N; x++)
  {
      InFuturo2 = In[x+1];
      In[x] = LFactor2*(InActual2*temp + InFuturo2 + InAnterior2);
      InAnterior2 = InActual2;
      InActual2 = InFuturo2;
  }
  
}

void SimulationStepMenor16 ( REAL * __restrict In, REAL L, int N, REAL bot, REAL Factor)
{
  REAL temp = 1/L - 2.0f;
  REAL InAnterior = bot;
  REAL InActual = In[0];
  REAL InFuturo;
  REAL LFactor = L*Factor;
    
  for (int x=0; x<N; x++)
  {
    InFuturo = In[x+1];
    In[x] = LFactor*(InActual*temp + InFuturo + InAnterior);
    InAnterior = InActual;
    InActual = InFuturo;
  }
}


void CopyVector ( REAL *In, REAL *Out, int N, REAL bot )
{
  int x;
  Out[0] = bot;
  for (x=0; x<N+1; x++)
    Out[x+1] = In[x];
}


REAL RandomFactor ( int K )
{
  int temp;

  temp = myRandom();
  int max = temp;
  int min = temp;
  
  for (int x=1; x<K; x++)
  {
   temp = myRandom();
   if (temp>max) max = temp;
   if (temp<min) min = temp;
  }

  return (REAL)(max-min)/(REAL)(max+min);
}


int main(int argc, char **argv)
{
  int  x, t, N=10000000, T=1000;
  REAL L= 0.123456789, L2, S;
  REAL *U1, *U2;
  REAL bot;
  int diff = N%16;
  int diffCiclos = T%2;
  seed = 12345;
  int N10 = N/10;
  
  
  if (argc>1) { T = atoi(argv[1]); }   // get  first command line parameter
  if (argc>2) { N = atoi(argv[2]); }   // get second command line parameter
  if (argc>3) { L = atof(argv[3]); }   // get  third command line parameter
  if (argc>4) { seed = atof(argv[4]);} // get fourth command line parameter
 
  if (N < 1 || L >= 0.5) {
    printf("arguments: T N L (T: steps, N: vector size, L < 0.5)\n");
    return 1;
  }

  U1 = malloc ( sizeof(*U1)*(N+1) );
  U2 = malloc ( sizeof(*U2)*(N+2) );
  if (!U1 || !U2) { printf("Cannot allocate vectors\n"); exit(1); }
  
  // initialize temperatures at time t=0  
  REAL t1, t2, t3;
  t1= 0.1234; t2 = -0.9456; t3 = 0.9789;
  t1= t1>1.3?   0.1234: t1+0.00567;
  t2= t2>1.8?  -0.9456: t2+0.00987;
  t3= t3<0.007? 0.9789: t3-0.00321;
  for (x=0; x<=N; x++)
  {
    U1[x] = t1*t2*t3;
    t1= t1>1.3?   0.1234: t1+0.00567;
    t2= t2>1.8?  -0.9456: t2+0.00987;
    t3= t3<0.007? 0.9789: t3-0.00321;
  }
 
  // initialize fixed boundary conditions on U1
  {
    bot = 1.2345678901;
    U1[N]= 1.2345678901;
  }
  
 printf("Challenge #4: Simulate %d steps on 1-D vector of %d elements with L=%1.10e\n", T, N, L);

  if(N >= 16)
  {
    if(diffCiclos != 0)
    {
      REAL Factor = RandomFactor(N10);
      SimulationStep ( U1, L, N, bot, diff, Factor ); 
    }
    
      

    for (int t=diffCiclos; t<T; t+=2)
    {  // loop on time
      REAL Factor1 = RandomFactor(N10);
      REAL Factor2 = RandomFactor(N10);
      SimulationStep2 ( U1, L, N, bot, diff, Factor1, Factor2 ); 
      
    }
  }
  
  else
  {
      for (int t=0; t<T; t++)
      {  // loop on time
        REAL Factor = RandomFactor(N10);
        SimulationStepMenor16 ( U1, L, N, bot, Factor ); 
      }
  }


  CopyVector(U1, U2, N, bot);
  free(U1); 
  
  printCheck(U2,N);
  free(U2);
}
