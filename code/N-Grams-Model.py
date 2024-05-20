import pickle
from collections import Counter
import math
import re
import os
import string
# https://github.com/dinkarjuyal/language-identification/blob/master/lang%2Bidentify.ipynb

def create_n_grams(lang, num, k): #returns top k n-grams according to frequency
    lang = " ".join(lang)
    words = re.sub('['+string.punctuation+']', '', lang) #  punctuation removed
    words = words.lower()
    words = re.sub('\s+', ' ', words).strip() # replaces multiple spaces, newline tabs with a single space
    words = words.replace(' ','_')# so that we can visualise spaces easily
    grams = {}
    #print (words)
    for i in range(len(words)-num):
        temp = words[i:i+num]
        if temp in grams:
            grams[temp] += 1
        else:
            grams[temp] = 1
    sum_freq = len(words) - num + 1
    for key in grams.keys():
        red = 1 # reduction factor equal 1 if no '_' is present
        if '_' in key: red = 2
        grams[key] = round(math.log(grams[key] / (red * sum_freq)), 3) #normalizing by dividing by total no of n-grams for that corpus and taking log                                             
    grams = sorted(grams.items(), key= lambda x : x[1], reverse = True) 
    #print (grams)
    final_grams = [] # contains a list of top k n-grams in a given language 
    log_probs = [] # contains logprobs corresponding to each n-gram
    for i in range(len(grams)):
        final_grams.append(grams[i][0])
        log_probs.append(grams[i][1])
    return final_grams, log_probs

# Calculate scores
def matching_score_2(test_grams, grams_list, n): # n helps us know whether it is bigram, trigram or quadgram
    dist = {lang: 0 for lang in lang_list} # distance corresponding to each language
    for gram in test_grams[0]:
        for lang in grams_list.keys():
            idx_2 = test_grams[0].index(gram)
            if gram in n_grams[n][lang][0] : 
                idx = n_grams[n][lang][0].index(gram)
                dist[lang] += abs(n_grams[n][lang][1][idx] - test_grams[1][idx_2])
            else: # gram is not present in that language's corpus
                dist[lang] += abs(test_grams[1][idx_2])
                # penalty term
    return dist 


def language_identify_2(file_address, st): # argument 'st' denotes whether you are uploading a file or directly copying text
    test_bigrams = []
    test_trigrams = []
    test_quadgrams = []
    test_file = []
    if st == 'file': # If you are copying a file address
        temp = file_address
        with open(temp, 'r', errors = 'ignore') as fname: # some characters throw an error with 'utf-8'
            file_address = fname.read()
    #print (file_address) 
    test_bigrams = create_n_grams(file_address, 2, k)
    test_trigrams = create_n_grams(file_address, 3, k)
    test_quadgrams = create_n_grams(file_address, 4, k)
    bi_dist = matching_score_2(test_bigrams, bi_grams, 2) 
    tri_dist = matching_score_2(test_trigrams, tri_grams, 3)
    quad_dist = matching_score_2(test_quadgrams, quad_grams, 4) 
    #print (bi_dist, tri_dist)
    final_dist = {}
    for lang in bi_dist.keys():
        final_dist[lang] =bi_dist[lang] + tri_dist[lang] + quad_dist[lang]
    sum_dist = 1
    for dist in final_dist.values():
        sum_dist += dist
    for lang in final_dist.keys():
        final_dist[lang] /= sum_dist
    dist_list = sorted(final_dist.items(), key= lambda x:x[1])     
    #print (dist_list)    
    # print ('Predicted language :' + dist_list[0][0] + '\n')
    return dist_list[0][0]

def save_model(model, filename):
    """
    Save a model to a file using pickle.

    Args:
    model: The model to save.
    filename (str): The filename to save the model to.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """
    Load a model from a file using pickle.

    Args:
    filename (str): The filename to load the model from.

    Returns:
    The loaded model.
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def filter_sentences(text_lines, min_length=5):
    filtered_sentences = []
    for sentence in text_lines:
        # Remove leading and trailing whitespace
        text = str(sentence.strip())
        text  = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = " ".join([st for st in text.split(" ") if st != ''])
        # Check if the sentence is empty or too short
        
        if len(text.split(' ')) >= min_length:
                    # Remove unusual characters using regular expressions  
                    #print(text.split(' ')) 
                    filtered_sentences.append(text)
        else:
            continue
    
    return filtered_sentences  

def evaluate(path_to_data):
        for item in os.listdir(path_to_data):
            # Construct the full path to the current item
            item_path = os.path.join(path_to_data, item)
            # Check if the current item is a directory
            if os.path.isdir(item_path):
                # Loop through the contents of the subfolder
                for sub_item in os.listdir(item_path):
                    # Construct the full path to the sub-item
                    sub_item_path = os.path.join(item_path, sub_item)

                    # Check if the sub-item is a file or directory
                    if os.path.isdir(sub_item_path):
                        lang = item_path.split("/")[-1]
                        files_manager = {}
                        for file_name in os.listdir(sub_item_path):
                                file_path = os.path.join(sub_item_path, file_name)

                                # Check if the current item is a file
                                if os.path.isfile(file_path):
                                            #print("File:", file_path)
                                            if 'train' in file_name:
                                                    files_manager['train_file'] = file_path
                                            elif 'dev' in file_name:
                                                    files_manager['dev_file'] = file_path
                                            elif 'test' in file_name:
                                                    files_manager['test_file'] = file_path  
                                            else:
                                                    print("This file is unknow")                              

                        # Dev
                        if  files_manager['test_file']:    
                                    corpus   = open(files_manager['test_file'], "r")
                                    corpus = corpus.readlines()
                                    for line in corpus:
                                                 l_identified = language_identify_2(line, 'string')
                                                 if l_identified == lang:
                                                            evals[lang][0] +=1
                                    evals[lang][1] = len(corpus)

                        # Dev
                        if  files_manager['dev_file']:    
                                    corpus   = open(files_manager['dev_file'], "r")
                                    corpus = corpus.readlines()
                                    for line in corpus:
                                                 l_identified = language_identify_2(line, 'string')
                                                 if l_identified == lang:
                                                            dev_evals[lang][0] +=1
                                    dev_evals[lang][1] = len(corpus)            
        return evals


# Example usage
# Example text data
# train_data_root_path = "./Train_test_split/"
eval_data_root_path = "../Create_N_gram_train_data/Train_test_split/"
train_data_root_path = "../../../../../ext_data/thapelo/NCHLT_Only/Vuk_Merged_NCHLT/"
models_root_path = './All-Grams-All-data-Models/'
bi_sorted        = "../../../../../ext_data/thapelo/CTEXT_Work/AlignedGov/All-Grams-All-data-Alinged/"
n = [2,3,4]  # Here we are choosing bigrams,trigrams and quadgrams; change this value to get n-grams with a particular n
k = 50 # Decides how many top n-grams will be used for calculating the distance metric
if __name__ == "__main__":
        bi_grams = {}
        tri_grams = {}
        quad_grams = {}
        lang_list = []
        for item in os.listdir(train_data_root_path):
            # Construct the full path to the current item
            item_path = os.path.join(train_data_root_path , item)
            # Check if the current item is a directory
            if os.path.isdir(item_path):
                print("-------------------------------------------------------------")
                print("Generating N-gram For language: ", item_path.split("/")[-1])

                # Loop through the contents of the subfolder
                for sub_item in os.listdir(item_path):
                    # Construct the full path to the sub-item
                    sub_item_path = os.path.join(item_path, sub_item)

                    # Check if the sub-item is a file or directory
                    if os.path.isdir(sub_item_path):
                        language = item_path.split("/")[-1]
                        lang_list.append(language)
                        files_manager = {}
                        for file_name in os.listdir(sub_item_path):
                                file_path = os.path.join(sub_item_path, file_name)

                                # Check if the current item is a file
                                if os.path.isfile(file_path):
                                            #print("File:", file_path)
                                            if 'train' in file_name:
                                                    files_manager['train_file'] = file_path
                                            elif 'dev' in file_name:
                                                    files_manager['dev_file'] = file_path
                                            elif 'test' in file_name:
                                                    files_manager['test_file'] = file_path  
                                            else:
                                                    print("This file is unknow")                              

                        # Train
                        if  files_manager['train_file']:    
                                    corpus   = open(files_manager['train_file'], "r")
                                    corpus = corpus.read()
                                    bi_grams[language] = create_n_grams(corpus, n[0], k)
                                    tri_grams[language] = create_n_grams(corpus, n[1], k)
                                    quad_grams[language] = create_n_grams(corpus, n[2], k)
                                            

            else:
                                    lang = item.split("_")[0]
                                    print("-------------------------------------------------------------")
                                    print("Generating N-gram For language: ", lang)
                                    lang_list.append(lang)
                                    corpus   = open(item_path, "r")
                                    corpus = corpus.read()
                                     
                                    bi_grams[lang] = create_n_grams(corpus, n[0], k)
                                    tri_grams[lang] = create_n_grams(corpus, n[1], k)
                                    quad_grams[lang] = create_n_grams(corpus, n[2], k)
        # save models
        n_grams = {2 : bi_grams, 3 : tri_grams, 4 : quad_grams}
        save_model(bi_grams,  models_root_path  +'bigram_model.pkl')
        save_model(tri_grams,  models_root_path +'trigram_model.pkl')
        save_model(quad_grams,  models_root_path +'quadgram_model.pkl')
        save_model(n_grams,  models_root_path  +'all_gram_model.pkl')
        evals = {lang:[0, 0] for lang in lang_list}
        dev_evals = {lang:[0, 0] for lang in lang_list}
        evaluation_scores = evaluate(eval_data_root_path)
        with open('./N_2_3_4_v_pl_nc_test_gram_evaluation_doc.txt', 'w') as e_file:
                                #    e_file.write(f'For {item} : model correclty predicted {values[1]/values[1]*100}% of {values[0]} sentences \n')
                                e_file.write(str(list(evals.keys())) + ' ||| ' + str([values[0]/values[1]*100 for values in list(evals.values())]))

        with open('./N_2_3_4_v_pl_nc_dev_gram_evaluation_doc.txt', 'w') as e_file:
                                #    e_file.write(f'For {item} : model correclty predicted {values[1]/values[1]*100}% of {values[0]} sentences \n')
                                e_file.write(str(list(dev_evals.keys())) + ' ||| ' + str([values[0]/values[1]*100 for values in list(dev_evals.values())]))                           

        # Test sentences
        ctext_root_folder = '../../../../../ext_data/thapelo/CTEXT_Work/AlignedGov/data_extracted/'
        for root, dirs, files in os.walk(ctext_root_folder):
                for file in files:
                        # Construct the full path to the file
                        text_path = os.path.join(root, file)
                        
                        # Open the file
                        with open(text_path, 'r') as f:
                            # Read the contents of the file
                            sentences = f.readlines()

                        # filter sentences
                        filtered_sentences = filter_sentences(sentences)

                        # Identify language for each sentence
                        for sentence in filtered_sentences:
                            # identified_language = identify_language(sentence, language_models)
                            if len("".join(filter(lambda x: not x.isdigit(), sentence)).lower().split()) > 3:
                                        identified_language = language_identify_2("".join(filter(lambda x: not x.isdigit(), sentence)).lower(), 'string')
                                        if identified_language == 'nso':
                                                with open(bi_sorted + 'n_gram_filtered_nso.txt', 'a') as n_file:
                                                                n_file.write(sentence + '\n')
                                        elif  identified_language == 'tsn':
                                            with open(bi_sorted + 'n_gram_filtered_tsn.txt', 'a') as tsn_file:
                                                                tsn_file.write(sentence + '\n') 
                                        elif identified_language == "sot":
                                            with open(bi_sorted + 'n_gram_filtered_sot.txt', 'a') as n_file:
                                                                n_file.write(sentence + '\n')
                                        elif identified_language == 'xho':
                                                with open(bi_sorted + 'n_gram_filtered_xho.txt', 'a') as n_file:
                                                                n_file.write(sentence + '\n')   
                                        elif identified_language == 'zul':
                                            with open(bi_sorted + 'n_gram_filtered_zul.txt', 'a') as n_file:
                                                                n_file.write(sentence + '\n')   
                                        elif identified_language == "ssw":
                                            with open(bi_sorted + 'n_gram_filtered_ssw.txt', 'a') as n_file:
                                                                n_file.write(sentence + '\n')   
                                        elif identified_language == "ven":
                                            with open(bi_sorted + 'n_gram_filtered_ven.txt', 'a') as n_file:
                                                                n_file.write(sentence + '\n') 
                                        elif identified_language == 'tso':
                                                with open(bi_sorted + 'n_gram_filtered_tso.txt', 'a') as n_file:
                                                                n_file.write(sentence + '\n')   
                                        elif identified_language == 'af':
                                                with open(bi_sorted + 'n_gram_filtered_af.txt', 'a') as n_file:
                                                                n_file.write(sentence + '\n')                                                                                                                                   
                                        elif identified_language == 'nbl':
                                            with open(bi_sorted + 'n_gram_filtered_nbl.txt', 'a') as n_file:
                                                                n_file.write(sentence + '\n')
                                        elif identified_language == 'eng':
                                            with open(bi_sorted + 'n_gram_filtered_en.txt', 'a') as n_file:
                                                                n_file.write(sentence + '\n')  
                                                                    
                                        else:
                                            # print("Language Unknown...") 
                                            continue 
