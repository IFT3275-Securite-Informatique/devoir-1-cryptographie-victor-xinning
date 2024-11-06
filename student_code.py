# Victor Yuyang Zhang (20247653)
# Xin Ning Xu (xxxxxxx)

# Solution a la question 2

import requests
from collections import Counter

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
  i = 0
  text_length = len(text)

  # Go through the entire text
  while i < text_length:

    # Verify a pair of caracters
    if i + 1 < text_length:
      pair = text[i] + text[i + 1]
      if pair in symbols_dict:
        symbols_dict[pair] += 1  # Increment the counter for that pair if present in the dict
        i += 2              # Evaluate the next pair
        continue

    # Verify a single caracter
    if text[i] in symbols_dict:
      symbols_dict[text[i]] += 1  # Increment the counter for that symbol if present in the dict
    i += 1

  return symbols_dict


# Function that tabulates the frequency of the encoded symbols within a cryptogram
def tabulate_encoded_symbols(split_cryptogram, symbols_dict):
  for symbol in split_cryptogram:
    if symbol in symbols_dict:
      symbols_dict[symbol] += 1
  return symbols_dict


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
  cryptogram_counter = tabulate_encoded_symbols(split_cryptogram, cryptogram_counter)
  text_counter = tabulate_symbols_in_text(text, text_counter)

  # Sort the values of the counter in descending order
  sorted_cryptogram_counter = sorted(cryptogram_counter.items(), key=lambda x: x[1], reverse=True)
  sorted_text_counter = sorted(text_counter.items(), key=lambda x: x[1], reverse=True)

  # Create a "statistical" key based on the words frenquency where (k,v)=(crypted_symbol,text_symbol)
  statistical_key = {}
  for i in range(len(sorted_cryptogram_counter)):
      crypted_symbol, _ = sorted_cryptogram_counter[i]
      text_symbol, _ = sorted_text_counter[i]
      statistical_key[crypted_symbol] = text_symbol

  # Decode the cryptogram using the statistical key
  M = []
  for crypted_symbol in split_cryptogram:
    M.append(statistical_key[crypted_symbol])

  # Convert the list of symbols into a string
  M = ''.join(M)
  
  return M