#include <stdio.h>
static int BITS_PER_DATA_ENTRY = 5;



int main() {
    printf("Hello, World!\n");
    return 0;
}
/*
 Script to binarize data in C
 - Data is read from csv
 - csv is iterated through line by line
    - for each line, take each element
        - for each element, convert to float, and get a binary representation
        - add last N bits from each binary representation to an array
    - add array of binarized elements to array of binarized lines



*/