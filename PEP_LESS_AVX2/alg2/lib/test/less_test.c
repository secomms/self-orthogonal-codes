#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fq_arith.h"
#include "codes.h"
#include "LESS.h"
#include "rng.h"
#include "api.h"

#include "test_helpers.c"

#define GRN "\e[0;32m"
#define WHT "\e[0;37m"

#ifdef N_pad
#define NN N_pad
#else
#define NN N
#endif

/// taken from the nist package.
void fprintBstr(FILE *fp, char *S, unsigned char *A, unsigned long long L) {
    unsigned long long  i;
    fprintf(fp, "%s", S);
    for ( i=0; i<L; i++ )
        fprintf(fp, "%02X", A[i]);
    if ( L == 0 )
        fprintf(fp, "00");
    fprintf(fp, "\n");
}

/* Exhaustive testing of inverses mod Q */
void inverse_mod_tester(void){
    uint32_t value[Q-1];
    uint32_t inverse[Q-1];
    for(uint32_t i=1; i <= Q-1; i++){
        value[i-1] = i;
        inverse[i-1] = fq_inv_non_ct(i);
    }
    int all_ok = 1;
    for(uint32_t i=1; i <= Q-1; i++){
        if((value[i-1]*inverse[i-1]) % Q !=1){
           printf("%u*%u=%u\n",
                  value[i-1],
                  inverse[i-1],
                  (value[i-1]*inverse[i-1])%Q);
           all_ok = 0;
        }
    }
    if (all_ok){
        puts("All inverses on F_q ok");
    }
}

/*
 *
 */
void rref_gen_byte_compress_tester(void){
     generator_mat_t G = {0}, Gcheck;
     uint8_t G_compressed [RREF_MAT_PACKEDBYTES];
     uint8_t is_pivot_column[NN];
     uint8_t was_pivot_column[NN] = {0};

     /* randomly generate a non-singular G */
     do {
         generator_rnd(&G);
         memset(is_pivot_column,0,sizeof(is_pivot_column));
     } while ( generator_RREF_pivot_reuse(&G,is_pivot_column,was_pivot_column,K) == 0);

     memcpy(&Gcheck,&G, sizeof(G));
     compress_rref(G_compressed,&G,is_pivot_column);
     generator_rnd(&G); /* fill with garbage to elicit faults */
     expand_to_rref(&G,G_compressed,is_pivot_column);

    if( memcmp( &Gcheck,&G,sizeof(generator_mat_t)) !=0 ){
        printf("Generator SF byte compression: ko\n");
       fprintf(stderr," Comp-decomp\n");
       generator_pretty_print_name("G",&G);

       fprintf(stderr,"is_pivot = \n [ ");
       for(uint64_t x=0;x < N ;x++){fprintf(stderr," %d ",is_pivot_column[x]); }
       fprintf(stderr,"]\n");

       fprintf(stderr," \n\n\n\n\n\n\n\n\nReference\n");
       generator_pretty_print_name("Gcheck",&Gcheck);
    } else {
        printf("Generator SF compression: ok\n");
    }
}

/*
 *
 */
void info(void){
    fprintf(stderr,"Code parameters: n= %d, k= %d, q=%d\n", N,K,Q);
    fprintf(stderr,"num. keypairs = %d\n",NUM_KEYPAIRS);
    fprintf(stderr,"Fixed weight challenge vector: %d rounds, weight %d \n",T,W);
    fprintf(stderr,"Private key: %luB\n", sizeof(prikey_t));
    fprintf(stderr,"Public key %luB, %.3f kiB\n", sizeof(pubkey_t), ((float) sizeof(pubkey_t))/1024);
    fprintf(stderr,"Signature: %luB, %.3f kiB\n", sizeof(sign_t), ((float) sizeof(sign_t))/1024);

}


#define NUMBER_OF_TESTS 1
#define MLEN 160

/* returns 1 if the test is successful, 0 otherwise */
int LESS_sign_verify_test_multiple(void){
    unsigned long long smlen = 0, mlen1;
    unsigned char *m  = (unsigned char *)calloc(MLEN, sizeof(unsigned char));
    unsigned char *sm = (unsigned char *)calloc(MLEN+CRYPTO_BYTES, sizeof(unsigned char));
    unsigned char       pk[CRYPTO_PUBLICKEYBYTES] = {0}, sk[CRYPTO_SECRETKEYBYTES] = {0};

    int ret = 0;
    for (size_t i = 0; i < NUMBER_OF_TESTS; ++i) {
        const uint32_t msg_len = (uint32_t)rand() % MLEN;
        randombytes(m, msg_len);

        memset(pk,0,CRYPTO_PUBLICKEYBYTES);
        memset(sk,0,CRYPTO_SECRETKEYBYTES);

        int ret_val;
        if ((ret_val = crypto_sign_keypair(pk, sk)) != 0) {
            printf("crypto_sign_keypair returned <%d>\n", ret_val);
            return -1;
        }

        fprintBstr(stdout, "pk = ", pk, CRYPTO_PUBLICKEYBYTES);
        fprintBstr(stdout, "sk = ", sk, CRYPTO_SECRETKEYBYTES);

        if ( (ret_val = crypto_sign(sm, &smlen, m, msg_len, sk)) != 0) {
            printf("crypto_sign returned <%d>\n", ret_val);
            return -1;
        }
        fprintBstr(stdout, "sm = ", sm, smlen);
        if ( (ret_val = crypto_sign_open(m, &mlen1, sm, smlen, pk)) != 0) {
            printf("crypto_sign_open returned <%d>\n", ret_val);
            //return -1;
            printf("Failed: %zu\n", i);
            continue;
        }

        if(mlen1 != msg_len) {
            printf("crypto_sign_open wrong length\n");
            return -1;
        }
        printf("OK: %zu\n", i);

        ret |= ret_val;
    }

    return ret;
}


int LESS_sign_verify_test_KAT(void) {
    uint8_t seed[48] = {0};
    uint8_t m[48] = {0};

    // init_randombytes(seed, 48);
    initialize_csprng(&platform_csprng_state,
                      (const unsigned char *)seed,
                      48);

    const uint32_t mlen = sizeof(m);
    unsigned long long smlen = 0, mlen1;
    unsigned char *m1 = (unsigned char *)calloc(mlen+CRYPTO_BYTES, sizeof(unsigned char));
    unsigned char *sm = (unsigned char *)calloc(mlen+CRYPTO_BYTES, sizeof(unsigned char));
    unsigned char pk[CRYPTO_PUBLICKEYBYTES] = {0}, sk[CRYPTO_SECRETKEYBYTES] = {0};

    int ret_val;
    if ((ret_val = crypto_sign_keypair(pk, sk)) != 0) {
        printf("crypto_sign_keypair returned <%d>\n", ret_val);
        return -1;
    }
    fprintBstr(stdout, "pk = ", pk, CRYPTO_PUBLICKEYBYTES);
    fprintBstr(stdout, "sk = ", sk, CRYPTO_SECRETKEYBYTES);

    if ( (ret_val = crypto_sign(sm, &smlen, m, mlen, sk)) != 0) {
        printf("crypto_sign returned <%d>\n", ret_val);
        return -1;
    }

    fprintBstr(stdout, "sm = ", sm, smlen);
    if ( (ret_val = crypto_sign_open(m1, &mlen1, sm, smlen, pk)) != 0) {
        printf("crypto_sign_open returned <%d>\n", ret_val);
        return -1;
    }

    free(m1);
    free(sm);
    printf("all good\n");
    return 0;
}

void test_key_recovery(void){
    uint8_t is_pivot_column[N] = {0};
    for(int i = 0; i < 10; i++){
        unsigned char seed[SEED_LENGTH_BYTES];
        randombytes(seed, SEED_LENGTH_BYTES);
        FQ_ELEM compressed[RREF_AO_BYTES] = {0};
        FQ_ELEM packed[RREF_AO_PACKEDBYTES] = {0};
        FQ_ELEM A_orig[K][K_pad] = {0};
        FQ_ELEM A_rec[K][K_pad] = {0};
        sample_antiorthogonal(A_orig,seed);

        FQ_ELEM extra_vars[K_pad] = {0};
        uint64_t bitstring[BITSTRING_LEN] = {0};
        compress_self_orthogonal(compressed,A_orig,extra_vars,bitstring);
        compress_rref_ao(packed,compressed,is_pivot_column);
        expand_rref_ao(compressed,packed,is_pivot_column);
        recover_self_orthogonal(A_rec,compressed,extra_vars,bitstring);

        for(int i=0; i<K; i++){
            for(int j=0; j<K_pad; j++){
                if(A_rec[i][j] != A_orig[i][j]){
                    printf("Recovery failed in position %u,%u.\n",i,j);
                    printf("%u != %u \n",A_rec[i][j],A_orig[i][j]);
                }
            }
        }
    }

}


void test_sample_antiorthogonal(){
    uint8_t error = 0;
    for(int i = 0; i<100; i++){
        unsigned char compressed_sk[PRIVATE_KEY_SEED_LENGTH_BYTES];
        randombytes(compressed_sk, PRIVATE_KEY_SEED_LENGTH_BYTES);
        SHAKE_STATE_STRUCT sk_shake_state;
        initialize_csprng(&sk_shake_state, compressed_sk, PRIVATE_KEY_SEED_LENGTH_BYTES);
        unsigned char G_seed[SEED_LENGTH_BYTES] = {0};
        csprng_randombytes(G_seed, SEED_LENGTH_BYTES, &sk_shake_state);
        FQ_ELEM A[K][K_pad] = {0};
        sample_antiorthogonal(A,G_seed);
        FQ_ELEM row_by_col;
        for(uint16_t i=0; i < K; i++){
            for(uint16_t j=0; j < K; j++){
                row_by_col = 0;
                for(uint16_t k=0; k<K; k++){
                    row_by_col = fq_add(row_by_col,fq_mul(A[i][k],A[j][k]));
                }
                if(row_by_col != 0 && i!=j){
                    printf("FAIL! Rows %u and %u are not orthogonal, inner_prod is %u!\n", i,j, row_by_col);
                    error = 1;
                }else if(row_by_col != 126 && i==j){
                    printf("FAIL! Row %u is not self-antiorthogonal, inner_prod is %u\n",i,row_by_col);
                    error = 1;
                }
            }
        }
    }
    if(!error){
        printf("Sample antiorthogonal: OK\n");
    }
}

void test_inner_prod(void){

    FQ_ELEM c[K_pad] = {0};
    uint8_t true_in_prod ;
    uint16_t in_prod;

    uint16_t error_ctr = 0;
    
    for(int run = 0; run < 128; run++){ 
        rand_range_q_elements(c,K);

        true_in_prod = 0;
        for(int i=0; i<K; i++){
            true_in_prod = (true_in_prod + (c[i]*c[i]))%127;
        }

        in_prod = superfast_inner_prod(c);

        if(true_in_prod != in_prod){
            printf("Inner prod differs %u != %u\n", in_prod, true_in_prod);
            error_ctr++;
        }
    }

    if(error_ctr){
        printf("inner_prod: %u out of %u went bad!\n",error_ctr,128);
    }else{
        printf("Inner prod: OK!\n");
    }


}

void test_row_mat_mult(void){

    uint16_t error = 0;
    for(int sample = 0; sample < 5; sample++){

        FQ_ELEM out[K_pad] = {0};
        FQ_ELEM out_avx2[K_pad] = {0};
        FQ_ELEM c[K_pad] = {0};
        rand_range_q_elements(c,K);
        FQ_ELEM G[K][K_pad] = {0};

        for(int i=0; i<K; i++){
            rand_range_q_elements(G[i],K);
        }

        for(int i=0; i<K; i++){
            for(int j=0; j<K; j++){
                out[i] = (out[i] + c[j]*G[j][i]) % 127;
            }
        }

        FQ_ELEM G_T[K][K_pad] = {0}; 

        for(int i=0; i<K; i++){
            for(int j=0; j<K; j++){
                G_T[i][j] = G[j][i];
            }
        }
        superfast_row_mat_mult(out_avx2,c,G_T,K,K);

        uint16_t error_ctr = 0;
        for(int i=0; i<K; i++){
            if(out_avx2[i] != out[i]){
                printf("Error in vector by matrix multiplication! %i %u != %u\n", i, out_avx2[i], out[i]);
                error_ctr++;
                error++;
            }
        }

        if(error_ctr){
            printf("Row Mat Mult: %u positions out of %u are wrong!\n", error_ctr,K);
        }
    }
    if(!error){
        printf("Row Mat Mult: OK\n");
    }
}

void test_anti_normalize(){
    FQ_ELEM c[K_pad] = {0};
    int counter = 0;
    int samples = 11000;
    for(int i=0; i<samples; i++){
        rand_range_q_elements(c, K);
        FQ_ELEM i_prod;
        inner_prod(&i_prod,c);
        anti_normalize(c);
        inner_prod(&i_prod,c);
        if(i_prod == 126) counter++;
    }
    printf("Counter antinormalization: %i/%i (%0.2f%)\n",counter,samples,(float)(counter*100)/samples); // -> 50%
}


void test_alg1(void){
    for(int i = 0; i < 100; i++){
        unsigned char seed[SEED_LENGTH_BYTES];
        randombytes(seed, SEED_LENGTH_BYTES);
        FQ_ELEM A_orig[K][K_pad] = {0};
        FQ_ELEM A[K][K_pad] = {0};
        FQ_ELEM A_rec[K][K_pad] = {0};
        sample_antiorthogonal(A_orig,seed);



        memcpy(A,A_orig,K*K_pad);

        for(uint16_t i=0; i<K; i++){
            for(uint16_t j=0; j<(i); j++){
                A[i][j] = 0;
            }
        }

        recover_self_orthogonal_alg1(A_rec,A);

        for(int i=0; i<K; i++){
            for(int j=0; j<K_pad; j++){
                if(A_rec[i][j] != A_orig[i][j]){
                    printf("Recovery failed in position %u,%u.\n",i,j);
                    printf("%u != %u \n",A_rec[i][j],A_orig[i][j]);
                }
            }
        }
    }

}



#define NUM_TEST_ITERATIONS 10
int main(int argc, char* argv[]){
    (void)argc;
    (void)argv;
    //test_inner_prod();
    //test_row_mat_mult();
    //test_anti_normalize();
    //test_sample_antiorthogonal();
    //test_key_recovery();
    //LESS_sign_verify_test_multiple();
    test_alg1();
    return 0;
}
