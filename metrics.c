#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

// implementazione in C delle metriche
// ho N come numero di sample, k come numero di batch
// y_true e y_pred sono matrici di dimensione (N, C) dove C è il numero di classi, array uint8_t
// tp, fp, tn, fn sono array uint64_t di dimensione C di shape K 


// questa è la 'firma' della funzione, prende dei puntatori a memoria (degli array) e rimpie gli gli array tp, tn, fp, fn
void metrics_count(const uint8_t* y_pred,const uint8_t* y_true, size_t N,size_t K, uint64_t* tp,uint64_t* tn, uint64_t* fp, uint64_t* fn){

 // popolo gli array con il valore 0
 for (size_t k=0; k < K; k++){
    tp[k]=fp[k]=fn[k]=tn[k]=0;

 }

 for (size_t sample=0; sample < N; sample++){ // per ogni sample
    size_t base_index = sample*K;
    for (size_t elem=0; elem < K; elem ++){

        if (y_pred[base_index + elem] == 1 && y_true[base_index + elem] == 1) {tp[elem]++;}
        else if (y_pred[base_index + elem] == 1 && y_true[base_index + elem] == 0) {fp[elem]++;}
        else if (y_pred[base_index + elem] == 0 && y_true[base_index + elem] == 0) {tn[elem]++;}
        else if (y_pred[base_index + elem] == 0 && y_true[base_index + elem] == 1) {fn[elem]++;}


    }
}
}











