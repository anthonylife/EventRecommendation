//Copyinright [2014] [Wei Zhang]

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
// Date: 2014/7/20                                               //
// Co-Factorization Machine for Event Recommendation.            //
///////////////////////////////////////////////////////////////////

#pragma once

#include "../utils.hpp"

#define MIN_CNT 1e-5

using namespace std;

class COFM{
    ////*****************Hyperparameter Settings****************//
    int niters;                                                 //
    int inner_niters;                                           //
    int nsample;                                                //
                                                                //
    int K;                                                      //
    double s_lr;                                                //
    double g_lr;                                                //
    double reg_u;                                               //
    double reg_e;                                               //
    int n_words;                                                //
    double alpha;                                               //
    //////////////////////////////////////////////////////////////

    vector<Event*>* total_events;
    vector<Event*>* tr_events;
    vector<Event*>* va_events;
    vector<Event*>* te_events;
    vector<hash_set<int>*>* puser_tr_events;

    vector<PAIR*>* tr_pairs;
    vector<PAIR*>* va_pairs;
    vector<PAIR*>* te_pairs;

    map<string, int> user_ids;
    map<int, string> ruser_ids;
    map<string, int> word_ids;
    map<int, string> rword_ids;
    map<string, int> event_ids;
    map<int, string> revent_ids;
    map<string, int> type_ids;
    map<int, string> rtype_ids;
    
    int n_users;
    int n_events;
    double * W;
    double ** theta_user;
    double ** theta_event;
    double ** beta_word;

    bool restart_tag;
    char* eventinfo_path;
    char* trdata_path;
    char* vadata_path;
    char* tedata_path;
    char* model_path;

public:
    void paraInit(){
        niters = 20;
        nsample = 5;
        K = 40;
        s_lr = 0.01;
        g_lr = 0.001;
        reg_u = 0.01;
        reg_w = 0.01;
        alpha = 1;
        n_words = 8000;

        total_events = new vector<Event*>();
        tr_events = new vector<Event*>();
        va_events = new vector<Event*>();
        te_events = new vector<Event*>();
        puser_tr_events = new vector<hash_set<int>*>();
        tr_pairs = new vector<Pair*>();
        va_pairs = new vector<Pair*>();
        te_pairs = new vector<Pair*>();
        W = NULL;
        theta_user = NULL;
        theta_word = NULL;

        restart_tag = false;
        eventinfo_path = NULL;
        trdata_path = NULL;
        vadata_path = NULL;
        tedata_path = NULL;
        model_path = NULL;
    }
    
    COFM(char * eventinfo_path, char * trdata_path, char * vadata_path,
            char * tedata_path, char * model_path, int tr_method,
            bool restart_tag) {
        paraInit();
        this->restart_tag = restart_tag;
        this->eventinfo_path = eventinfo_path;
        this->trdata_path = trdata_path;
        this->vadata_path = vadata_path;
        this->tedata_path = tedata_path;
        this->model_path = model_path;
        this->tr_method = tr_method;

        printf("Loading preference pairs.\n");
        utils::loadPrefPairs(trdata_path, tr_pairs, total_events, tr_events,
                user_ids, ruser_ids, event_ids, revent_ids);
        utils::loadPrefPairs(vadata_path, va_pairs, total_events, va_events,
                user_ids, ruser_ids, event_ids, revent_ids);
        utils::loadPrefPairs(tedata_path, te_pairs, total_events, te_events,
                user_ids, ruser_ids, event_ids, revent_ids);
        n_users = (int)user_ids.size();
        n_events = (int)total_events->size();
        printf("Number of users: %d.\n", n_users);
        printf("Number of events: %d.\n", n_events);

        printf("Loading event information.\n");
        utils::loadEventInfo(eventinfo_path, total_events, event_ids,
                revent_ids, word_ids, rword_ids, type_ids, rtype_ids,
                n_words);
        printf("Word Num: %d\n", n_words);
        int zero_intro = 0;
        for (vector<Event*>::iterator it=total_events->begin();
                it!=total_events->end(); it++) {
            if ((*it)->intro.size()==0)
                zero_intro++;
        }
        cout << "Number of zero intro: "<< zero_intro << endl;

        printf("Getting event list for each user in training data.\n");
        utils::getUserEventList(tr_pairs, puser_tr_events,event_ids,n_users);

        /*for (vector<hash_set<int>*>::iterator it=puser_tr_events->begin();
                it!=puser_tr_events->end(); it++) {
            for (hash_set<int>::iterator it1=(*it)->begin();
                    it1!=(*it)->end(); it1++)
                printf("%d ", *it1);
            printf("\n");
            utils::pause();
        }*/

        if (restart_tag) {
            printf("Random initialization model parameters.\n");
            factorInit();
        } else {
            printf("Loading trained model parameters.\n");
            loadModelPara();
        }
    }

    ~COFM() {
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
        if (total_events) {
            for (vector<Event*>::iterator it=total_events->begin();
                    it!=total_events->end(); it++)
                delete (*it);
            delete total_events;
        }
        if (tr_events) {
            delete tr_events;
        }
        if (va_events) {
            delete va_events;
        }
        if (te_events) {
            delete te_events;
        }
        if (puser_tr_events) {
            for (vector<hash_set<int>*>::iterator it=puser_tr_events->begin();
                    it!=puser_tr_events->end(); it++)
                delete (*it);
            delete puser_tr_events;
        }

        user_ids.clear();
        map<string, int>(user_ids).swap(user_ids);
        ruser_ids.clear();
        map<int, string>(ruser_ids).swap(ruser_ids);
        word_ids.clear();
        map<string, int>(word_ids).swap(word_ids);
        rword_ids.clear();
        map<int, string>(rword_ids).swap(rword_ids);
        event_ids.clear();
        map<string, int>(event_ids).swap(event_ids);
        revent_ids.clear();
        map<int, string>(revent_ids).swap(revent_ids);
        type_ids.clear();
        map<string, int>(type_ids).swap(type_ids);
        rtype_ids.clear();
        map<int, string>(rtype_ids).swap(rtype_ids);
    }

    void factorInit() {
        int num_para = (n_users+n_words+n_events)*K;

        W = new double[num_para];
        int ind = 0;
        theta_user = new double*[n_users];
        for (int u=0; u<n_users; u++) {
            theta_user[u] = W + ind;
            utils::muldimGaussrand(theta_user[u], K, 0, 1);
            //utils::muldimZero(theta_user[u], K);
            ind += K;
        }
        theta_event = new double*[n_events];
        for (int e=0; e<n_events; e++) {
            theta_event[e] = W + ind;
            utils::muldimGaussrand(theta_event[e], K, 0, 1);
            //utils::muldimZero(theta_user[u], K);
            ind += K;
        }
        theta_word = new double*[K];
        for (int k=0; k<K; k++) {
            theta_word[k] = W + ind;
            for (int w=0; w<n_words; w++)
                theta_word[k][w] = 1.0/n_words;
            ind += n_words;
        }
    }

    void train() {
        if (tr_method == 1)
            // batch learning for topic word matrix using BPR loss
            bprBatchLearn();
        else if (tr_method == 2)
            // mini-batch learning for topic word matrix using BPR loss
            bprMiniBatchLearn();
        else if (tr_method == 3)
            // mini-batch learning for topic word matrix using WARP loss
            warpMiniBatchLearn();
        else {
            printf("Invalid choice of learning method!\n");
            exit(1);
        }
    }

    void bprBatchLearn() {
        int finished_num, ind;
        int uid, wid;
        double p_val, n_val, wn_ep, logit_loss, normal;
        double cur_train, cur_valid, best_valid;
        double * gtheta_user = new double[K];
        double * gptheta_event = new double[K];
        double * gntheta_event = new double[K];
        double ** gbeta_word = NULL;
        Event *p_event = NULL, *n_event=NULL;

        gbeta_word = new double*[K];
        for (int k=0; k<K; k++) {
            gbeta_word[k] = new double[n_words];
            memset(gbeta_word[k], 0.0, sizeof(double)*n_words);
        }    

        cur_valid = 0.0;
        best_valid = 0.5;
        timeval start_t, end_t;
        for (int i=0; i<niters; i++) {
            finished_num = 0;
            random_shuffle(tr_pairs->begin(), tr_pairs->end());
            utils::tic(start_t);
            
            /// Learn user and event latent factors (SGD)
            for (vector<Pair*>::iterator it=tr_pairs->begin();
                    it!=tr_pairs->end(); it++) {
                uid = (*it)->uid;
                p_event = (*it)->event;
                if (p_event->intro.size() == 0)
                    continue;
                p_val = utils::dot(theta_user[uid], theta_event[p_event->eventid], K);
                random_shuffle(tr_events->begin(), tr_events->end());
                ind = 0;
                for (vector<Event*>::iterator it1=tr_events->begin();
                        it1!=tr_events->end(); it1++) {
                    if ((*puser_tr_events)[uid]->find((*it1)->eventid) != (*puser_tr_events)[uid]->end() || (*it1)->intro.size() == 0)
                        continue;
                    ind++;
                    n_event = *it1;
                    memset(gtheta_user, 0.0, sizeof(double)*K);
                    memset(gptheta_event, 0.0, sizeof(double)*K);
                    memset(gntheta_event, 0.0, sizeof(double)*K);
                    n_val = utils::dot(theta_user[uid], theta_event[n_event->eventid], K);
                    logit_loss = utils::logitLoss(p_val-n_val);
                    // compute gradients from user behavior part
                    for (int k=0; k<K; k++) {
                        gtheta_user[k] = logit_loss*(theta_event[p_event->eventid][k]-theta_event[(*it1)->eventid][k]);
                        gptheta_event[k] = logit_loss*theta_user[uid][k];
                        gntheta_event[k] = -logit_loss*theta_user[uid][k];
                    }
                    // compute gradients from content part
                    for (vector<Cword>::iterator it2=p_event->wordcnt.begin(); it2!=p_event->wordcnt.end(); it2++) {
                        wn_ep = 0.0;
                        for (int k=0; k<K; k++)
                            wn_ep = theta_event[p_event->eventid][k]*beta_word[k][it2->wid];
                        for (int k=0; k<K; k++) 
                            gptheta_event[k] += (it2->wcnt/(wn_ep+MIN_CNT)-1)*beta_word[k][it2->wid];
                    }
                    for (vector<Cword>::iterator it2=n_event->wordcnt.begin();it2!=n_event->wordcnt.end(); it2++) {
                        for (int k=0; k<K; k++)
                            wn_ep = theta_event[n_event->eventid][k]*beta_word[k][it2->wid];
                        for (int k=0; k<K; k++) 
                            gntheta_event[k] += (it2->wcnt/(wn_ep+MIN_CNT)-1)*beta_word[k][it2->wid];
                    }
                    // update parameters
                    for (int k=0; k<K; k++) {
                        theta_user[uid][k] += s_lr*(gtheta_user[k]-reg_u*theta_user[uid][k]);
                        theta_event[p_event->eventid][k] += s_lr*(gptheta_event[k]-reg_e*theta_event[p_event->eventid][k]);
                        theta_event[p_event->eventid][k] = utils::max(0.0, theta_event[p_event->eventid][k]);
                        theta_event[n_event->eventid][k] += s_lr*(gntheta_event[k]-reg_e*theta_event[n_event->eventid][k]);
                        theta_event[n_event->eventid][k] = utils::max(0.0, theta_event[n_event->eventid][k]);
                    }
                    if (ind == nsample)
                        break;
                }
                if (finished_num%100000 == 0) {
                    printf("\rCurrent Iteration: %d, Finished Training Pair Num: %d.", i+1, finished_num);
                    utils::toc(start_t, end_t, false);
                    fflush(stdout);
                    utils::tic(start_t);
                }
            }
            
            /// Learn new event latent factor (SGD) 
            for (vector<Event*>::iterator it=te_events->begin();
                    it!=te_events->end(); it++) {
                memset(gptheta_event, 0.0, sizeof(double)*K);
                // compute gradients
                for (vector<Cword>::iterator it1=(*it)->wordcnt.begin();
                        it1!=(*it)->wordcnt.end(); it1++) {
                    for (int k=0; k<K; k++)
                        wn_ep = theta_event[(*it)->eventid][k]*beta_word[k][it1->wid];
                    for (int k=0; k<K; k++) 
                        gptheta_event[k] += (it1->wcnt/(wn_ep+MIN_CNT)-1)*beta_word[k][it1->wid];
                }
                // update parameters
                for (int k=0; k<K; k++){
                    theta_event[(*it)->eventid][k] += s_lr*(gptheta_event[k]-reg_e*theta_event[(*it)->eventid][k]);
                    theta_event[(*it)->eventid][k] = utils::max(0.0, theta_event[(*it)->eventid][k]);
                }
            }

            /// Learn dictionary (PGD)
            for (int k=0; k<K; k++)
                memset(gbeta_word[k], 0.0, sizeof(double)*n_words);
            // compute gradients from content
            for (vector<Event*>::iterator it=tr_events->begin();
                    it!=tr_events->end(); it++) {
                for (vector<Cword>::iterator it1=(*it)->wordcnt.begin();
                        it1!=(*it)->wordcnt.end(); it1++) {
                    for (int k=0; k<K; k++)
                        wn_ep = theta_event[p_event->eventid][k]*beta_word[k][it1->wid];
                    for (int k=0; k<K; k++)                    
                        gbeta_word[k][it1->wid] += (it1->wcnt/(wn_ep+MIN_CNT)-1)*theta_event[(*it)->eventid][k];
                }
            }
            // update dictionary
            for (int k=0; k<K; k++) {
                for (int w=0; w<n_words; w++)
                    beta_word[k][w] += g_lr*beta_word[k][w];
                utils::project_beta1(beta_word[k], n_words, 0.0);
            }

            //utils::toc(start_t, end_t, true);
            //utils::pause();
            evaluation(cur_train, cur_valid);
            printf("\nTrain AUC: %f, Validaton AUC: %f!\n", cur_train, cur_valid);
            if (cur_valid > best_valid)
                best_valid = cur_valid;
        }
        saveModelPara();
        delete[] gtheta_user;
        delete[] gptheta_event;
        delete[] gntheta_event;
        for (int k=0; k<K; k++)
            delete[] gbeta_word[k];
        delete[] gbeta_word;
    }

    void recommendation(char* submission_path) {
        int uid, finished_num;
        double score, normal;
        string uname, ename;
        double* sword_factor = new double[K];
        vector<int>* consumers = new vector<int>();
        vector<Rateval>* inter_result = NULL;
        Rateval* rateval = NULL;
   
        /// preprocessing before starting recommendation
        utils::getConsumers(tedata_path, consumers, user_ids);

        printf("\nStart recommendation!\n");
        finished_num = 0;
        ofstream* out = utils::ofstream_(submission_path);
        for (vector<int>::iterator it=consumers->begin();
                it!=consumers->end(); it++) {
            uid = *it;
            inter_result = new vector<Rateval>();
            for (vector<Event*>::iterator it1=te_events->begin();
                    it1!=te_events->end(); it1++) {
                ename = (*it1)->event_nameid;
                memset(sword_factor, 0.0, sizeof(double)*K);
                for (vector<int>::iterator it2=(*it1)->intro.begin();
                        it2!=(*it1)->intro.end(); it2++)
                    for (int k=0; k<K; k++)
                        sword_factor[k] += theta_word[*it2][k];
                if ((*it1)->intro.size() == 0)
                    score = 0.0;
                else
                    normal = sqrt((*it1)->intro.size());
                    score = 0.0;
                    for (int k=0; k<K; k++)
                        score += theta_user[uid][k]*sword_factor[k]/normal;
                rateval = new Rateval();
                rateval->id = ename;
                rateval->score = score;
                inter_result->push_back(*rateval);
                delete rateval;
            }
            sort(inter_result->begin(), inter_result->end(), utils::greaterCmp);
            *out << ruser_ids[uid];
            for (vector<Rateval>::iterator it1=inter_result->begin();
                    it1!=inter_result->end(); it1++) {
                *out << "," << it1->id;
            }
            *out << endl;
            finished_num++;
            printf("\rFinished Recommendation Pairs: %d", finished_num);
            // release memory
            delete inter_result;
        }

        out->close();
        delete out;
        delete[] sword_factor;
        delete consumers;
    }

    void evaluation(double & train, double & test) {
        int correct_num = 0, total_num = 0, finished_num=0;
        int uid, wid;
        double score, p_val, n_val;
        timeval start_t, end_t;
        Event * event;

        vector<vector<double>*>* user_score = new vector<vector<double>*>(); 
        for (int u=0; u<n_users; u++) {
            vector<double>* event_score = new vector<double>();
            user_score->push_back(event_score);
        }

        for (int u=0; u<n_users; u++) {
            for (int e=0; e<n_events; e++) {
                event = (*total_events)[e];
                if (event->intro.size() == 0)
                    score = 0.0;
                else
                    score = utils::dot(theta_user[u], theta_event[e], K);
                (*user_score)[u]->push_back(score);
            }
        }

        correct_num = 0;
        total_num = 0;
        finished_num = 0;
        for (vector<Pair*>::iterator it=tr_pairs->begin();
                it!=tr_pairs->end(); it++) {
            uid = (*it)->uid;
            event = (*it)->event;
            if ((*it)->event->intro.size() == 0)
                continue;
            p_val = (*(*user_score)[uid])[event->eventid];
            for (vector<Event*>::iterator it1=tr_events->begin();
                    it1!=tr_events->end(); it1++) {
                if ((*puser_tr_events)[uid]->find((*it1)->eventid)
                        != (*puser_tr_events)[uid]->end())
                    continue;
                if ((*it1)->intro.size() == 0)
                    continue;
                total_num++;
                n_val = (*(*user_score)[uid])[(*it1)->eventid];
                if (p_val > n_val)
                    correct_num++;
            }
            finished_num++;
            if (finished_num%100 == 0) {
                printf("\rFinished Evaluation Pair Num: %d.", finished_num);
                utils::toc(start_t, end_t, false);
                fflush(stdout);
                utils::tic(start_t);
            }
        }
        printf("\n");
        train = 1.0*correct_num/total_num;
        //cout << "correct_num: " << correct_num << endl;
        //cout << "total_num: " << total_num << endl;
        //utils::pause();

        correct_num = 0;
        total_num = 0;
        finished_num = 0;
        for (vector<Pair*>::iterator it=te_pairs->begin();
                it!=te_pairs->end(); it++) {
            uid = (*it)->uid;
            event = (*it)->event;
            if ((*it)->event->intro.size() == 0)
                continue;
            p_val = (*(*user_score)[uid])[event->eventid];
            for (vector<Event*>::iterator it1=tr_events->begin();
                    it1!=tr_events->end(); it1++) {
                if (*it1 == event)
                    continue;
                if ((*it1)->intro.size() == 0)
                    continue;
                total_num++;
                n_val = (*(*user_score)[uid])[(*it1)->eventid];
                if (p_val > n_val)
                    correct_num++;
            }
            finished_num++;
            if (finished_num%100 == 0) {
                printf("\rFinished Evaluation Pair Num: %d.", finished_num);
                utils::toc(start_t, end_t, false);
                fflush(stdout);
                utils::tic(start_t);
            }
        }
        printf("\n");
        test = 1.0*correct_num/total_num;
        
        for (int u=0; u<n_users; u++) {
            delete (*user_score)[uid];
        }
        delete user_score;
    }
   
    void loadModelPara() {
        int num_para = (n_users+n_words+n_events)*K;

        W = new double[num_para];
        FILE* f = utils::fopen_(model_path, "r");
        utils::fread_(W, sizeof(double), num_para, f);
        delete f;

        int ind = 0;
        theta_user = new double*[n_users];
        for (int u=0; u<n_users; u++) {
            theta_user[u] = W + ind;
            ind += K;
        }
        theta_event = new double*[n_events];
        for (int e=0; e<n_events; e++) {
            theta_event[w] = W + ind;
            ind += K;
        }
        beta_word = new double*[K];
        for (int k=0; k<K; k++) {
            beta_word[k] = W + ind;
            ind += n_words;
        }
    }
    
    void saveModelPara() {
        int num_para = (n_users+n_words+n_events)*K;
        FILE* f = utils::fopen_(model_path, "w");
        fwrite(W, sizeof(double), num_para, f);
    }
};


