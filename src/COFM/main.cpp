//Copyright [2014] [Wei Zhang]

//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//http://www.apache.org/licenses/LICENSE-2.0
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

///////////////////////////////////////////////////////////////////
// Date: 2014/7/12                                               //
// Running corresponding model (COFM)                            //
///////////////////////////////////////////////////////////////////


#include "model.hpp"

using namespace std;

string DATA_ROOT_PATH = "/home/anthonylife/Doctor/Code/MyPaperCode/EventRecommendation/data/";
string TRAIN_FILE[2] = {"doubanTrainEventBeijing.csv", "doubanTrainEventShanghai.csv"};
string VALIDATION_FILE[2] = {"doubanValiEventBeijing.csv", "doubanValiEventShanghai.csv"};
string TEST_FILE[2] = {"doubanTestEventBeijing.csv", "doubanTestEventShanghai.csv"};
string EVENT_FILE[2] = {"doubanEventBeijing.csv", "doubanEventShanghai.csv"};
string RESULT_ROOT_PATH = "/home/anthonylife/Doctor/Code/MyPaperCode/EventRecommendation/result/";
string SUBMISSION_PATH[2] = {RESULT_ROOT_PATH + string("sigir12.result1"), RESULT_ROOT_PATH + string("sigir12.result2")};


int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}


int main(int argc, char **argv) {
    int i;
    int a=0;
    char *b=NULL;
    if (argc == 1) {
        printf("COFM v 0.1a\n");
        printf("\tExamples:\n");
        printf("./run -d 0(1) -r True(False)\n\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-d", argc, argv)) > 0) a = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-r", argc, argv)) > 0) b = argv[i + 1];
    if (a!=0 && a!=1) {
        printf("Invalid choice of dataset!\n");
        exit(1);
    }
   
    string submission_path = SUBMISSION_PATH[a];
    string model_path = "./factor.model";
    string event_path = DATA_ROOT_PATH + EVENT_FILE[a];
    string trdata_path = DATA_ROOT_PATH + TRAIN_FILE[a];
    string vadata_path = DATA_ROOT_PATH + VALIDATION_FILE[a];
    string tedata_path = DATA_ROOT_PATH + TEST_FILE[a];
    //int data_num = a;
    bool restart_tag;
    
    if (strcmp(b, (char *)"True") == 0)
        restart_tag = true;
    else if (strcmp(b, (char *)"False") == 0)
        restart_tag = false;
    else {
        printf("Invalid input of para -r\n");
        exit(1);
    }
   
    timeval start_t, end_t;
    utils::tic(start_t);
    
    COFM *cofm = new COFM((char *)event_path.c_str(),
                          (char *)trdata_path.c_str(),
                          (char *)vadata_path.c_str(),
                          (char *)tedata_path.c_str(),
                          (char *)model_path.c_str(),
                          restart_tag);
    if (restart_tag)
        cofm->train();
    cofm->recommendation((char *)submission_path.c_str());
    
    utils::toc(start_t, end_t);

    return 0;
}
