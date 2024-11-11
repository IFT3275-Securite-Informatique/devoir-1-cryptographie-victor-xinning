# Victor Yuyang Zhang (20247653)
# Xin Ning Xu (xxxxxxx)

# Solution a la question 2

import math
import random as rnd
import numpy as np
import requests
import re
import string
from collections import Counter, defaultdict

#=============================================================================================#
# Functions defined in (or inspired from) the homework instructions ==========================#
#=============================================================================================#

# Function that takes a text (string) and returns a list of its decomposition
# into pairs of caracters
def cut_string_into_pairs(text):
  pairs = []
  for i in range(0, len(text) - 1, 2):
    pairs.append(text[i:i + 2])
  if len(text) % 2 != 0:
    pairs.append(text[-1] + '_')  # Add a placeholder if the string has an odd number of characters
  return pairs

# Loads a text from the url given in argument. Returns a string
def load_text_from_web(url):
  try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.text
  except requests.exceptions.RequestException as e:
    print(f"An error occurred while loading the text: {e}")
    return None

# From a given text, returns a list of the its unique caracters + bicaracters
def obtain_symbols(text):
  caracteres = list(set(list(text)))  # Set of unique characters in the text
  nb_bicaracteres = 256 - len(caracteres)
  bicaracteres = [item for item, _ in Counter(cut_string_into_pairs(text)).most_common(nb_bicaracteres)]
  symboles = caracteres + bicaracteres

  return symboles


#=============================================================================================#
# Auxiliary functions ========================================================================#
#=============================================================================================#

# Function that takes a list of symbols and creates a dictionnary (k,v) = (symbol,counter)
def create_symbol_counter(symbols):
    symbol_counter = {}  # Dictionary to store the count of each symbol
    for symbol in symbols:
        symbol_counter[symbol] = 0  # Counter = 0
    return symbol_counter


# Function that seperates a cryptogram (string) into a list of encoded symbols
def separate_cryptogram(C):
  separation = []
  for i in range(0, len(C), 8):
    separation.append(C[i:i+8])
  return separation


# Function that tabulates the frequency of symbols / pairs of symbols for a text (string)
def tabulate_symbols_in_text(text, symbols_dict):

  bigram_symbols_dict = {}
  trigram_symbols_dict = {}
  num_bigrams = 0               # For normalization
  num_trigrams = 0              # For normalization

  text_length = len(text)
  i = 0
  second_previous_symbol = None
  previous_symbol = None

  # Go through the entire text
  while i < text_length:

    # Verify a pair of caracters
    if i + 1 < text_length:
      pair = text[i] + text[i + 1]
      if pair in symbols_dict:
        symbols_dict[pair] += 1  # Increment the counter for that pair if present in the dict
        i += 2              # Evaluate the next pair

        # Update the bigram and trigram
        if previous_symbol is not None:
          bigram_symbols_dict[(previous_symbol, pair)] = bigram_symbols_dict.get((previous_symbol, pair), 0) + 1
          num_bigrams += 1
        if second_previous_symbol is not None:
          trigram_symbols_dict[(second_previous_symbol, previous_symbol, pair)] = trigram_symbols_dict.get((second_previous_symbol, previous_symbol, pair), 0) + 1
          num_trigrams += 1

        # Update second_previous_symbol and previous_symbol
        second_previous_symbol = previous_symbol
        previous_symbol = pair

        continue

    # Verify a single caracter
    single = text[i]
    if single in symbols_dict:
      symbols_dict[single] += 1  # Increment the counter for that symbol if present in the dict

      # Update the bigram and trigram
      if previous_symbol is not None:
        bigram_symbols_dict[(previous_symbol, single)] = bigram_symbols_dict.get((previous_symbol, single), 0) + 1
        num_bigrams += 1
      if second_previous_symbol is not None:
        trigram_symbols_dict[(second_previous_symbol, previous_symbol, single)] = trigram_symbols_dict.get((second_previous_symbol, previous_symbol, single), 0) + 1
        num_trigrams += 1

      # Update second_previous_symbol and previous_symbol
      second_previous_symbol = previous_symbol
      previous_symbol = single

    i += 1
  
  # Normalize the values within the bigram and trigram dict
  bigram_symbols_dict = {k: v / num_bigrams for k, v in bigram_symbols_dict.items()}
  trigram_symbols_dict = {k: v / num_trigrams for k, v in trigram_symbols_dict.items()}

  return symbols_dict, bigram_symbols_dict, trigram_symbols_dict


# Function that tabulates the frequency of the encoded symbols within a cryptogram
def tabulate_encoded_symbols(split_cryptogram, symbols_dict):

  bigram_symbols_dict = {}
  trigram_symbols_dict = {}
  num_bigrams = 0               # For normalization
  num_trigrams = 0              # For normalization
  second_previous_symbol = None
  previous_symbol = None

  for symbol in split_cryptogram:
    if symbol in symbols_dict:
      symbols_dict[symbol] += 1

      # Update the bigram and trigram
      if previous_symbol is not None:
        bigram_symbols_dict[(previous_symbol, symbol)] = bigram_symbols_dict.get((previous_symbol, symbol), 0) + 1
        num_bigrams += 1
      if second_previous_symbol is not None:
        trigram_symbols_dict[(second_previous_symbol, previous_symbol, symbol)] = trigram_symbols_dict.get((second_previous_symbol, previous_symbol, symbol), 0) + 1
        num_trigrams += 1

      # Update second_previous_symbol and previous_symbol
      second_previous_symbol = previous_symbol
      previous_symbol = symbol

  # Normalize the values within the bigram and trigram dict
  bigram_symbols_dict = {k: v / num_bigrams for k, v in bigram_symbols_dict.items()}
  trigram_symbols_dict = {k: v / num_trigrams for k, v in trigram_symbols_dict.items()}

  return symbols_dict, bigram_symbols_dict, trigram_symbols_dict


# Scoring system using birams and trigrams
def compute_score(text, text_symbols_set, bigram_frequencies, trigram_frequencies):
  score = 0

  text_length = len(text)
  i = 0
  second_previous_symbol = None
  previous_symbol = None

  # Go through the entire text
  while i < text_length:

    # Verify a pair of caracters
    if i + 1 < text_length:
      pair = text[i] + text[i + 1]
      if pair in text_symbols_set:

        # Add to score if bigram or trigram is valid
        if previous_symbol is not None:
          bigram = (previous_symbol, pair)
          if bigram in bigram_frequencies:
            score += bigram_frequencies[bigram]

        if second_previous_symbol is not None:
          trigram = (second_previous_symbol, previous_symbol, pair)
          if trigram in trigram_frequencies:
            score += trigram_frequencies[trigram]

        # Update second_previous_symbol and previous_symbol
        second_previous_symbol = previous_symbol
        previous_symbol = pair

        i += 2              # Evaluate the next pair
        continue

    # Verify a single caracter
    single = text[i]
    if single in text_symbols_set:

      # Add to score if bigram or trigram is valid
      if previous_symbol is not None:
        bigram = (previous_symbol, single)
        if bigram in bigram_frequencies:
          score += bigram_frequencies[bigram]

      if second_previous_symbol is not None:
        trigram = (second_previous_symbol, previous_symbol, single)
        if trigram in trigram_frequencies:
          score += trigram_frequencies[trigram]

      # Update second_previous_symbol and previous_symbol
      second_previous_symbol = previous_symbol
      previous_symbol = single

    i += 1

  # idea: we can maybe add to score if the words belong in the french dictionnary? (no, takes too long)
  #words = text.split()
  #french_words = load_french_words()
  #for word in words:
  #
  #  word = clean_text(word)
  #  if is_word_in_dictionary(word, french_words):
  #    score += 0.2

  return score


# Function that optimizes the key, using the bigrams and trigrams to compute a score
def optimize_key(split_cryptogram, initial_key, bigram_frequencies, trigram_frequencies, ranked_symbols_list, text_symbols_set, iterations=1000):

  current_key = initial_key.copy()
  best_key = current_key.copy()

  sample_size = min(1500, len(split_cryptogram))

  decrypted_text = []
  for crypted_symbol in split_cryptogram[:sample_size]:
    decrypted_text.append(best_key[crypted_symbol])
  decrypted_text = ''.join(decrypted_text)

  best_score = compute_score(decrypted_text, text_symbols_set, bigram_frequencies, trigram_frequencies)

  for _ in range(iterations):

    # slightly randomize the key
    randomized_key = swap_symbols(ranked_symbols_list, current_key.copy(), 10)
    
    decrypted_text = []
    for crypted_symbol in split_cryptogram[:sample_size]:
      decrypted_text.append(randomized_key[crypted_symbol])
    decrypted_text = ''.join(decrypted_text)

    # Compute score of the randomized key
    score = compute_score(decrypted_text, text_symbols_set, bigram_frequencies, trigram_frequencies)

    # Keep track of the best key / score
    if score > best_score:
      best_score = score
      best_key = randomized_key.copy()
      current_key = randomized_key.copy()

  return best_key, best_score


# Function that randomly swaps two values while respecting a certain interval within a ranking of keys
def swap_symbols(ranked_symbols_list, mapping_dict, interval):

  # Choose a random index
  i = rnd.randint(0, len(ranked_symbols_list) - 1)

  # Get the interval of accepted swaps
  lower_bound = max(0, i - interval)
  upper_bound = min(len(ranked_symbols_list) - 1, i + interval)

  # Choose a random index within the interval
  j = rnd.randint(lower_bound, upper_bound)

  # Make sure i != j
  while i == j:
    j = rnd.randint(lower_bound, upper_bound)

  # Get the symbols
  symbol_i = ranked_symbols_list[i]
  symbol_j = ranked_symbols_list[j]

  # Swap the letters associated to the crypted symbols
  mapping_dict[symbol_i], mapping_dict[symbol_j] = mapping_dict[symbol_j], mapping_dict[symbol_i]

  return mapping_dict



#=============================================================================================#
# Code for partial brute force (not used, takes too long to run) =============================#
#=============================================================================================#

# Loads the french dictionary (source: https://github.com/mmai/chiensuperieur/blob/master/dictionnaires/liste.de.mots.francais.frgut.txt)
def load_french_words(url="https://raw.githubusercontent.com/mmai/chiensuperieur/refs/heads/master/dictionnaires/liste.de.mots.francais.frgut.txt"):
    try:
        response = requests.get(url)
        response.raise_for_status()
        words = response.text.splitlines()
        return set(words)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while loading the dictionary: {e}")
        return set()

# Returns true if the word is in the dictionnary
def is_word_in_dictionary(word, french_words):
    return word.lower() in french_words

def contains_whitespace(s):
    return any(c in s for c in [' ', '\n', '\r', '\t', '\ufeff'])

# Function that takes a string and removes all special caracters from it
def clean_text(text):
  # Split the text into words
  words = text.split()

  # Check if the first word contains an apostrophe
  if words and "'" in words[0]:
    # Remove all characters before the first apostrophe in the first word
    words[0] = re.sub(r".*'", "", words[0])

  # Join the words back into a single string
  text = ' '.join(words)

  # Remove special characters, keeping only letters and spaces
  text = re.sub(r"[^a-zA-Z\s'À-ÿ]", "", text)

  # Remove extra spaces
  text = re.sub(r"\s+", " ", text).strip()

  return text

# Function that brute forces a key while using a statistical key as a guideline
def partial_brute_force(split_cryptogram, split_cryptogram_pointer,
                text_symbols_ranking, all_text_symbols,
                statistical_key, french_words,
                next_word, current_mapping, error_counter):
  """
  logic behind the brute force:
    1. Start with the first 5 crypted symbols, and find their position within the sorted text_symbols
    2. with these 5 positions, we can suppose that the real value for that crypted symbole is within a +/- 5 position?
    3. We test all of the possible mapping, and for each of them, when we detect a " " or "\n", ... we check if the word is valid (in the french dictionnary)
    4. After these 5, we test the next 5, and we do that until we obtain a coherent text
    5. Bruh i hope this works

  :param split_cryptogram: a list of the encoded symbols
  :param statistical_key: a dictionnary (k,v) = (crypted_symbol,text_symbol)
  :param text_symbols_ranking: a list of the symbols from most frequent to least frequent
  :param french_words: a set of french words
  :return:
  """
  # End condition: when we reached the end of the cryptogram, we return the current_mapping (key)
  if split_cryptogram_pointer >= len(split_cryptogram):
    return True, current_mapping

  # Obtain the next crypted symbol to test
  crypted_symbol = split_cryptogram[split_cryptogram_pointer]

  # Loop until we obtain a crypted symbol that doesn't have an association yet in the current_mapping
  while crypted_symbol in current_mapping:

    next_word = next_word + current_mapping[crypted_symbol]

    # If we finished forming a word: verify in french dictionnary
    if contains_whitespace(next_word):

      word = clean_text(next_word)
      words = word.split()

      # If words contains letters and not only whitespace
      if words:

        # if the word obtained is NOT in the dictionnary, error_counter+=1 until we reach a max tolerated error of 100?
        if not is_word_in_dictionary(words[0], french_words):

          # We want to have at least 15 correct symbols before accepting errors
          if split_cryptogram_pointer < 15:
            return False, current_mapping

          error_counter += 1

          if error_counter > 100:
            return False, current_mapping

      # New word being formed
      next_word = ""
      if len(words) > 1:
        next_word = words[1]

    # <- end of if the word contains whitespace

    # Increment the pointer to get the next crypted symbol
    split_cryptogram_pointer += 1

    # End condition: when we reached the end of the cryptogram, we return the current_mapping (key)
    if split_cryptogram_pointer >= len(split_cryptogram):
      return True, current_mapping

    else: # get the next crypted symbol
      crypted_symbol = split_cryptogram[split_cryptogram_pointer]

  # <- end while (now we have a crypted_symbol that isn't mapped to a text symbol)

  # Get the relative ranking (index) of the crypted symbol
  relative_c_s_index = all_text_symbols.index(statistical_key[crypted_symbol]) / len(all_text_symbols)
  c_s_index = int(relative_c_s_index * len(text_symbols_ranking))

  # Verify all possible symbols from the vicinity (+/- 5) of the index
  for i in range(max(c_s_index - 5, 0), min(c_s_index + 5, len(text_symbols_ranking) - 1)):

    # Get the symbol
    symbol = text_symbols_ranking[i]

    # Construct the word
    current_word = next_word + symbol

    # If we finished forming a word: verify in french dictionnary
    if contains_whitespace(current_word):

      word = clean_text(current_word)
      words = word.split()

      # If words contains letters and not only whitespace
      if words:

        # if the word obtained is NOT in the dictionnary, error_counter+=1 until we reach a max tolerated error of 100?
        if not is_word_in_dictionary(words[0], french_words):

          # We want to have at least 15 correct symbols before accepting errors
          if split_cryptogram_pointer < 15:
            continue

          error_counter += 1

          if error_counter > 100:
            return False, current_mapping

      # New word being formed
      current_word = ""
      if len(words) > 1:
        current_word = words[1]

    # <- end of if the word contains whitespace

    # Increment the pointer to get the next crypted symbol
    split_cryptogram_pointer += 1

    # End condition: when we reached the end of the cryptogram, we return the current_mapping (key)
    if split_cryptogram_pointer >= len(split_cryptogram):
      current_mapping_copy = current_mapping.copy()
      current_mapping_copy[crypted_symbol] = symbol
      return True, current_mapping

    else: # else, we need to analyse the next crypted symbol (recursive call)

      # Make a copy of the text_symbols_ranking and remove the one used for the current symbol
      text_symbols_ranking_copy = text_symbols_ranking.copy()
      del text_symbols_ranking_copy[i]

      # Make a copy of the current_mapping and add an entry (crypted_symbol, symbol)
      current_mapping_copy = current_mapping.copy()
      current_mapping_copy[crypted_symbol] = symbol

      # Recursive call
      bool_result, final_mapping = brute_force(split_cryptogram, split_cryptogram_pointer,
                                               text_symbols_ranking_copy, all_text_symbols,
                                               statistical_key, french_words,
                                               current_word, current_mapping_copy, error_counter)

      if not bool_result:
        continue
      else:
        #print("returning 5")
        return True, final_mapping

  # <- end of for loop

  # return false
  return False, current_mapping

# <- end of brute_force function


#=============================================================================================#
# Main decryption function ===================================================================#
#=============================================================================================#

# Function that takes a cryptogram and returns its corresponding plain text using
# a statistical approach
def decrypt(C):

  # Load the text that will be used for the statistical counting
  url = "https://www.gutenberg.org/ebooks/13846.txt.utf-8"  # Example URL (replace with your desired URL)
  text = load_text_from_web(url)
  url = "https://www.gutenberg.org/ebooks/4650.txt.utf-8"  # Example URL (replace with your desired URL)
  text = text + load_text_from_web(url)

  # Seperate the cryptogram into a list of its encoded symbols
  split_cryptogram = separate_cryptogram(C)

  # Seperate the text into a list of its caracters (bicaracters)
  text_symbols = obtain_symbols(text)

  # Create a dictionnary for both lists of symbols
  cryptogram_counter = create_symbol_counter(list(set(split_cryptogram)))
  text_counter = create_symbol_counter(text_symbols)

  # Tabulate the symbols frenquency for the text and the cryptogram
  cryptogram_counter, bigram_cryptogram_counter, trigram_cryptogram_counter = tabulate_encoded_symbols(split_cryptogram, cryptogram_counter)
  text_counter, bigram_text_counter, trigram_text_counter = tabulate_symbols_in_text(text, text_counter)

  # Sort the values of the counter in descending order
  sorted_cryptogram_counter = sorted(cryptogram_counter.items(), key=lambda x: x[1], reverse=True)
  sorted_text_counter = sorted(text_counter.items(), key=lambda x: x[1], reverse=True)

  #print(sorted_text_counter)
  #print(len(sorted_text_counter))

  # Create a "statistical" key based on the words frenquency where (k,v)=(crypted_symbol,text_symbol)
  statistical_key = {}
  for i in range(len(sorted_cryptogram_counter)):
    crypted_symbol, _ = sorted_cryptogram_counter[i]
    text_symbol, _ = sorted_text_counter[i]
    statistical_key[crypted_symbol] = text_symbol

  ## Brute force
  #text_symbols_ranking = [symbol_and_count[0] for symbol_and_count in sorted_text_counter]
  #print (text_symbols_ranking)
  #print(set(split_cryptogram))
  #result, mapping = partial_brute_force(split_cryptogram, 0, text_symbols_ranking, text_symbols_ranking, statistical_key, load_french_words(), "", {}, 0)

  # Get the best key
  crypted_symbols_ranking = [symbol_and_count[0] for symbol_and_count in sorted_cryptogram_counter]
  text_symbols_set = set(text_symbols)
  best_key = None
  best_overall_score = 0

  for _ in range(5):
    key, score = optimize_key(split_cryptogram, statistical_key, bigram_text_counter, trigram_text_counter, crypted_symbols_ranking, text_symbols_set, 50000)
    
    # Keep the best score
    if score > best_overall_score:
      best_overall_score = score
      best_key = key

  # Decrypt using the best key
  M = []
  for crypted_symbol in split_cryptogram:
    M.append(best_key[crypted_symbol])

  # Convert the list of symbols into a string
  M = ''.join(M)

  return M
