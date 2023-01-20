#include <stdio.h>
#include <malloc.h>

static int BITS_PER_DATA_ENTRY = 5;

struct List{
    int length;
    int *list;
};

struct List_of_Lists{
    int length;
    struct List *list;
};

void Append_1D(struct List *list, int new_element){
    list->list = realloc(list->list, sizeof(&list->list)+sizeof(int));
    list->list[list->length] = new_element;
    list->length++;
}

void Append_2D(struct List_of_Lists *list, struct List new_element){
    list->list = realloc(list->list, sizeof(&list->list)+sizeof(new_element));
    list->list[list->length] = new_element;
    list->length++;
}

void printList1D(struct List mylist){
    for (int i = 0; i<mylist.length; i++){
        printf("%d", mylist.list[i]);
    }
    printf("\n");
}

void printList2D(struct List_of_Lists mylists){
    for (int i = 0; i<mylists.length; i++){
        printList1D(mylists.list[i]);
    }
    printf("\n");
}

int main() {
    printf("1D lists:\n");
    struct List myList;
    myList.list = malloc(1*sizeof(int));
    myList.list[0] = 0;
    myList.length = 1;

    Append_1D(&myList, 1);
    Append_1D(&myList, 2);
    Append_1D(&myList, 3);
    Append_1D(&myList, 4);
    printList1D(myList);

    struct List myList2;
    myList2.list = malloc(1*sizeof(int));
    myList2.list[0] = 5;
    myList2.length = 1;
    Append_1D(&myList2, 6);
    Append_1D(&myList2, 7);
    Append_1D(&myList2, 8);
    Append_1D(&myList2, 9);
    printList1D(myList2);

    struct List myList3;
    myList3.list = malloc(1*sizeof(int));
    myList3.list[0] = 10;
    myList3.length = 1;
    Append_1D(&myList3, 11);
    Append_1D(&myList3, 12);
    Append_1D(&myList3, 13);
    Append_1D(&myList3, 14);
    printList1D(myList3);


    printf("2D list:\n");
    struct List_of_Lists myLists;
    myLists.list = malloc(1*sizeof(myList));
    myLists.list[0] = myList;
    myLists.length = 1;

    Append_2D(&myLists,myList2);
    printList2D(myLists);
    Append_2D(&myLists, myList3);
    printList2D(myLists);
    printf("done printing!");

    /// Free allocated memory
    free(myList.list);
    free(myList2.list);
    free(myList3.list);
    free(myLists.list);
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