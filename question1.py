import math
import random as rnd
import numpy as np
import sympy
from sympy import factorint
import requests
from collections import Counter

# convert string to list of integer
def str_to_int_list(x):
  z = [ord(a) for a in x  ]
  for x in z:
    if x > 256:
      print(x)
      return False
  return z

# convert a strint to an integer
def str_to_int(x):
  x = str_to_int_list(x)
  if x == False:
    print("Le text n'est pas compatible!")
    return False

  res = 0
  for a in x:
    res = res * 256 + a
  i = 0
  res = ""
  for a in x:
    ci = "{:08b}".format(a )
    if len(ci)>8:
      print()
      print("long",a)
      print()
    res = res + ci
  res = eval("0b"+res)
  return res

# exponentiation modulaire
def modular_pow(base, exponent, modulus):
    result = 1
    base = base % modulus
    while exponent > 0:
        if (exponent % 2 == 1):
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    return result

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

# inverse multiplicatif de a modulo m
def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception("Pas d'inverse multiplicatif")
    else:
      return x % m

def decrypt(c):    
  N = int(input("Please enter your N: "))
  e = int(input('Please enter your e: '))

  mNumber = sympy.integer_nthroot(c, e)
  if mNumber[1] == True:
    if c == modular_pow(int(mNumber[0]), int(e), int(N)):
      M = int(mNumber[0]).to_bytes((int(mNumber[0]).bit_length() + 7) // 8, 'big').decode()
      print(M)
    else:
      raise Exception("Pas le bon message!")

  elif mNumber[1] == False:
    p = 10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069673
    q = 16072629107794009814226375735900027158421072175583004111656255825555265766874041837397975682235437871913920093763297202377807179285384710653976866362047862205901851662236346478131611907593556712816931273229569712475372911901098151338748315919115594371856794716529813251490644747478936580257043048672231
    phiN = (p-1)*(q-1)
    d = modinv(e, phiN)
    m = modular_pow(c, d, N)
    M = int(m).to_bytes((int(m).bit_length() + 7) // 8, 'big').decode()
    print(M)

decrypt(25782248377669919648522417068734999301629843637773352461224686415010617355125387994732992745416621651531340476546870510355165303752005023118034265203513423674356501046415839977013701924329378846764632894673783199644549307465659236628983151796254371046814548224159604302737470578495440769408253954186605567492864292071545926487199114612586510433943420051864924177673243381681206265372333749354089535394870714730204499162577825526329944896454450322256563485123081116679246715959621569603725379746870623049834475932535184196208270713675357873579469122917915887954980541308199688932248258654715380981800909)
  