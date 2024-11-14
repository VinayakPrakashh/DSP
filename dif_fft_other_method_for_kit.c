#include<stdio.h>
#include<math.h>
#include<complex.h>
#define PI 3.148
int i;
int main()
{   int z;
    printf("enter the value of N:- ");
    scanf("%d",&z);
    int N=z;
    double complex x[N],real,imag;
    printf("enter the  input values\n");
    for(i=0;i<N;i++)
    {
        printf("enter the element %d (real,img):",i);
        scanf("%lf,%lf",&real,&imag);
        x[i]=real+imag*I;
    }
    dif(x,N);
    //bit_reversal(x,N);
    print(x,N);
    ifft(x,N);
    bit_reversal(x,N);
    printifft(x,N);
    return 0;
}
void dif(double complex x[],int N)
{double complex even[N/2], odd[N/2];
    if(N==1)
        return;
    else
    {

        for(i=0;i<N/2;i++)
        {
            even[i]=x[i]+x[i+N/2];
            odd[i]=(x[i]-x[i+N/2])*cexp((-2*PI*I*i)/N);
        }
    }
    dif(even, N / 2);
    dif(odd, N / 2);
    for(i=0;i<N/2;i++)
        {
          x[2*i]    = even[i];
          x[2*i + 1]= odd[i];
        }

}
void bit_reversal(double complex x[],int N)
{  double complex even[N/2],odd[N/2];

    if(N<=1)
        return;
    else
    {

        for(i=0;i<N/2;i++)
        {
            even[i]=x[2*i];
            odd[i]=x[2*i + 1];
        }
    }
    bit_reversal(even,N/2);
    bit_reversal(odd,N/2);
    for(i=0;i<N/2;i++)
        {
          x[i+N/2]=odd[i] ;
          x[i]= even[i];
        }

}
void ifft(complex double x[], int N) {
    if (N == 1) return;

    complex double even[N/2];
    complex double odd[N/2];

    for (int i = 0; i < N / 2; i++) {
        even[i] = x[i] + x[i + N/2];
        odd[i] = (x[i] - x[i + N/2]) * cexp(2.0 * PI * I * i / N);
    }

    ifft(even, N / 2);
    ifft(odd, N / 2);

    for (int i = 0; i < N / 2; i++) {
        x[i] = even[i];
        x[i + N / 2] = odd[i];
    }

}

void print(double complex x[],int N)
{

    for(i=0;i<N;i++)
        printf("X[%d]= %0.2lf +i %0.2lf\n",i,creal(x[i]),cimag(x[i]));
}
void printifft(double complex x[],int N)
{

    for(i=0;i<N;i++)
        printf("X[%d]= %0.2lf +i %0.2lf\n",i,creal(x[i]/N),cimag(x[i]/N ));
}