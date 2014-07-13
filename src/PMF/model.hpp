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
// Probabilistic Matric Factorization with Pairwise Learning for //
//   implicit feedback data.                                     //
///////////////////////////////////////////////////////////////////

#pragma once

#include "../utils.hpp"

using namespace std;
using namespace __gnu_cxx;


struct PAIR{
    int uid;
    int oid;
};
typedef struct PAIR Pair;

class PMF{
    ////*****************Hyperparameter Settings****************//
    int niters;                                                 //
    int nsample;                                                //
                                                                //
    int K;                                                      //
    int lr;                                                     //
    double reg_u;                                               //
    double reg_o;                                               //
    //////////////////////////////////////////////////////////////
    
    vector<PAIR*>* tr_pairs;
    vector<PAIR*>* va_pairs;
    vector<PAIR*>* te_pairs;

    map<string, int> user_ids;
    map<int, string> ruser_ids;
    map<string, int> organizor_ids;
    map<int, string> rorganizor_ids;
    vector<int> organizors;

    vector<hash_set<int>*>* tr_interacted_organizors;
    vector<hash_set<int>*>* va_interacted_organizors;

    int n_users;
    int n_organizors;
    double * W;
    double ** theta_user;
    double ** theta_organizor;

    bool restart_tag;
    char* trdata_path;
    char* vadata_path;
    char* tedata_path;
    char* model_path;

public:
    void paraInit(){
        niters = 100;
        nsample = 5;
        K = 40;
        lr = 0.05;
        reg_u = 0.1;
        reg_o = 0.1;

        tr_pairs = NULL;
        va_pairs = NULL;
        te_pairs = NULL;
        W = NULL;
        theta_user = NULL;
        theta_organizor = NULL;
        tr_interacted_organizors = NULL;
        va_interacted_organizors = NULL;

        restart_tag = false;
        trdata_path = NULL;
        vadata_path = NULL;
        tedata_path = NULL;
        model_path = NULL;
    }
    
    PMF(char * trdata_path, char * vadata_path, char * tedata_path,
            char * model_path, bool restart_tag) {
        paraInit();
        this->restart_tag = restart_tag;
        this->trdata_path = trdata_path;
        this->vadata_path = vadata_path;
        this->tedata_path = tedata_path;
        this->model_path = model_path;

        printf("Get mapping relation about user and id.\n");
        utils::getMapOfUserId(trdata_path, user_ids, ruser_ids,
                organizor_ids, rorganizor_ids, n_users, n_organizors);
        utils::getMapOfUserId(vadata_path, user_ids, ruser_ids,
                organizor_ids, rorganizor_ids, n_users, n_organizors);
        utils::getMapOfUserId(tedata_path, user_ids, ruser_ids,
                organizor_ids, rorganizor_ids, n_users, n_organizors);
        printf("User: %ld, Organizor: %ld\n", user_ids.size(), organizor_ids.size());
        for (uint i=0; i<organizor_ids.size(); i++)
            organizors.push_back(i);
        printf("Loading preference pairs.\n");
        tr_pairs = loadPrefPairs(trdata_path);
        va_pairs = loadPrefPairs(vadata_path);
        te_pairs = loadPrefPairs(tedata_path);

        tr_interacted_organizors = getInteractedOrg(tr_pairs);
        va_interacted_organizors = getInteractedOrg(va_pairs);

        if (restart_tag) {
            printf("Random initialization model parameters.\n");
            factorInit();
        } else {
            printf("Loading trained model parameters.\n");
            loadModelPara();
        }
    }

    ~PMF() {
        if (tr_pairs) {
            for (vector<Pair*>::iterator it=tr_pairs->begin();
                    it!=tr_pairs->end(); it++)
                delete (*it);
            delete tr_pairs;
        }
        if (va_pairs) {
            for (vector<Pair*>::iterator it=va_pairs->begin();
                    it!=va_pairs->end(); it++)
                delete (*it);
            delete va_pairs;
        }
        if (te_pairs) {
            for (vector<Pair*>::iterator it=te_pairs->begin();
                    it!=te_pairs->end(); it++)
                delete (*it);
            delete te_pairs;
        }
        if (tr_interacted_organizors) {
            for (vector<hash_set<int>*>::iterator it=tr_interacted_organizors->begin(); it!=tr_interacted_organizors->end(); it++) {
                delete (*it);
            }
            delete tr_interacted_organizors;
        }
        if (va_interacted_organizors) {
            for (vector<hash_set<int>*>::iterator it=va_interacted_organizors->begin(); it!=va_interacted_organizors->end(); it++) {
                delete (*it);
            }
            delete va_interacted_organizors;
        }

        user_ids.clear();
        map<string, int>(user_ids).swap(user_ids);
        ruser_ids.clear();
        map<int, string>(ruser_ids).swap(ruser_ids);
        organizor_ids.clear();
        map<string, int>(organizor_ids).swap(organizor_ids);
        rorganizor_ids.clear();
        map<int, string>(rorganizor_ids).swap(rorganizor_ids);
    }

    void factorInit() {
        int num_para = (n_users+n_organizors)*K;

        W = new double[num_para];
        int ind = 0;
        theta_user = new double*[n_users];
        for (int u=0; u<n_users; u++) {
            theta_user[u] = W + ind;
            //utils::muldimGaussrand(theta_user[u], K);
            utils::muldimZero(theta_user[u], K);
            ind += K;
        }
        theta_organizor = new double*[n_organizors];
        for (int o=0; o<n_organizors; o++) {
            theta_organizor[o] = W + ind;
            //utils::muldimGaussrand(theta_organizor[o], K);
            utils::muldimZero(theta_organizor[o], K);
            ind += K;
        }
    }

    vector<PAIR*>* loadPrefPairs(data_path) {
        vector<string> parts;
        string line, uname, oname;
        vector<Pair*> pairs = new vector<Pair*>();
        
        ifstream* in = utils::ifstream_(data_path);
        while (getline(*in, line)) {
            parts = utils::split_str(line, ',');
            Pair * pair = new Pair();
            pair->uid = user_ids[parts[0]];
            pair->oid = organizor_ids[parts[1]];
            pairs->push_back(pair);
        }
        in->close();
        delete in;
        return pairs;
    }

    vector<hash_set<int>*>* getInteractedOrg(vector<Pair*>* pairs) {
        vector<hash_set<int>*>* interacted_organizors=new vector<hash_set<int>*>();
        for (int u=0; u<n_users; u++) {
            hash_set<int>* orgnizor_set = new hash_set<int>();
            interacted_organizors->push_back(orgnizor_set);
        }
        
        int uid, oid;
        for (vector<Pair*>::iterator it=pairs->begin();
                it!=pairs->end(); it++) {
            uid = (*it)->uid;
            oid = (*it)->oid;
            interacted_organizors[uid]->insert(oid);
        }
        return interacted_organizors;
    }


    void train() {
        int finished_num, ind;
        int uid, oid, coid;
        double * tuser_factor = new double[K];
        double cur_train, cur_valid, best_valid;
        
        cur_valid = 0.0;
        best_valid = 0.5;
        timeval start_t, end_t;
        for (int i=0; i<niters; i++) {
             finished_num = 0;
             random_shuffle(tr_pairs->begin(), tr_pairs->end());
             utils::tic(start_t);
             for (vector<Pair*>::iterator it=tr_pairs->begin();
                     it!=tr_pairs->end(); it++) {
                uid = (*it)->uid;
                oid = (*it)->oid;
                random_shuffle(organizors.begin(), organizors.end());
                ind = 0;
                for (vecotr<int>::iterator it1=organizors.begin();
                        it1!=organizors.end(); it1++) {
                    if ((*it1) == oid)
                        continue;

                    ind++;
                    p_val = utils::dot(theta_user[uid], theta_organizor[oid], K);
                    n_val = utils::dot(theta_user[uid], theta_organizor[*it1], K);
                    logit_loss = utils::logitLoss(p_val-n_val);
//#pragma omp parallel for
                    for (int k=0; k<K; k++) {
                        tuser_factor[j] = theta_user[uid][k]
                            + lr*(logit_loss*(theta_organizor[oid][k]
                                -theta_organizor[*it1][k])
                                -reg_u*theta_user[uid][k]);
                        theta_organizor[oid][k] = theta_organizor[oid][k]
                            + lr*(logit_loss*theta_user[uid][k]
                                -theta_organizor[oid][k]);
                        theta_organizor[*it1][k] = theta_organizor[*it1][k]
                            + lr*(-logit_loss*theta_user[uid][k]
                                -theta_organizor[*it1][k]);
                    }
                    memcpy(theta_user[uid], tuser_factor, K*sizeof(double));

                    if (ind == nsample)
                        break;
                }
                finished_num++;
                if (finished_num%10000 == 0) {
                    printf("\rCurrent Iteration: %d, Finished Training Pair Num: %d.", i+1, finished_num);
                    fflush(stdout);
                }
            }
            utils::toc(start_t, end_t, true);
            //utils::pause();
            evaluation(cur_train, cur_valid);
            if (cur_valid > best_valid)
                best_valid = cur_valid;
        }
        saveModelPara();
        delete tr_pairs; 
    }

    void recommendation(char* submisssion_path) {
        map<string, int> event_ids;;
        map<int, string> revent_ids;
        map<int, int>* event_organizor = new map<int, int>();
        vector<int>* consumers = new vector<int>();
   
        /// preprocessing before starting recommendation
        utils::getMapOfSpecificRegion(tedata_path, 1, event_ids, revent_ids);
        utils::getEventOrganizor(tedata_path, event_organizor, event_ids,
                organizor_ids);
        utils::getConsumers(tedata_path, consumers, user_ids);

        printf("\nStart recommendation!\n");
        

        event_ids.clear();
        map<string, int>(event_ids).swap(event_ids);
        revent_ids.clear();
        map<int, string>(revent_ids).swap(revent_ids);
        delete event_organizor;
    }

    double evaluation(double & train, double & valid) {
        int correct_num = 0, total_num = 0;
        int uid, oid, noid;
        double p_val, n_val;

        correct_num = 0;
        total_num = 0;
        for (vector<Pair*>::iterator it=tr_pairs->begin();
                it!=tr_pairs->end(); it++) {
            uid = (*it)->uid;
            oid = (*it)->oid;
            p_val = utils::dot(theta_user[uid], theta_organizor[oid], K);
            for (vector<int>::iterator it1=organizors.begin();
                    it1!=organizors.end(); it++) {
                if (tr_interacted_organizors->find(*it1)==tr_interacted_organizors->end()) {
                    total_num++;
                    n_val = utils::dot(theta_user[uid], theta_organizor[*it1], K);
                    if (p_val > n_val)
                        correct_num++;
                }
            }
        }
        train = 1.0*correct_num/total_num;

        correct_num = 0;
        total_num = 0;
        for (vector<Pair*>::iterator it=va_pairs->begin();
                it!=va_pairs->end(); it++) {
            uid = (*it)->uid;
            oid = (*it)->oid;
            p_val = utils::dot(theta_user[uid], theta_organizor[oid], K);
            for (vector<int>::iterator it1=organizors.begin();
                    it1!=organizors.end(); it++) {
                if (tr_interacted_organizors->find(*it1)==tr_interacted_organizors->end() && va_interacted_organizors->find(*it1)==va_interacted_organizors->end()) {
                    total_num++;
                    n_val = utils::dot(theta_user[uid], theta_organizor[*it1], K);
                    if (p_val > n_val)
                        correct_num++;
                }
            }
        }
        test = 1.0*correct_num/total_num;
    }
    
    void loadModelPara() {
        int num_para = (n_users+n_organizors)*K;

        W = new double[num_para];
        FILE* f = utils::fopen_(model_path, "r");
        utils::fread_(W, sizeof(double), num_para, f);
        fclose(f);
        delete f;

        int ind = 0;
        theta_user = new double*[n_users];
        for (int u=0; u<n_users; u++) {
            theta_user[u] = W + ind;
            ind += K;
        }
        theta_organizor = new double*[n_organizors];
        for (int o=0; o<n_organizors; o++) {
            theta_organizor[o] = W + ind;
            ind += K;
        }
    }
    
    void saveModelPara() {
        int num_para = (n_users+n_organizors)*K;
        FILE* f = utils::fopen_(model_path, "w");
        fwrite(W, sizeof(double), num_para, f);
    }
};


