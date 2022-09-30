# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:51:01 2022
1...array as numpy
2.derivative function
3.tensorflow

@author: wajee
"""
import numpy as np

def f(x):
    return 2*x
arr = [1,2,3,4]

#***************USINGB FOR retrieving the element from array**********
for element in arr:
    print(f(element))
    
#1**************Using NUMPY ************************
a_array = np.array([1,2,3,4])
print(f(a_array))

#2..***************USING derivative manually****************

def function(x):
    return 3*x**2
def der_function(x):
    return 6*x
print(der_function(3))


#2...*************USING DERIVATIVE FoRMULA************
def derivative_fun(x):
    h = 0.000001
    return (function(x-h)-function(x))/h
print(derivative_fun(3.0))

#3....***********USING TENSORFLOW*****************
import  tensorflow as tf
x = tf.Variable(3.0)

with tf.GradientTape() as tape: 
    y = function(x)
    
val = tape.gradient(y,x)
print(val)
#4....****************USING TENSORFLOW MULTIVARIABLE DEFRENTIAL_____PARTIAL DERIVATIVE**************
def function_partial(X,Y):
    return X**2 + Y**2
    
X = tf.Variable(-10.0)
Y = tf.Variable(-23.0)
with tf.GradientTape() as tape:
     z = function_partial(X,Y)
     print(z)
v = tape.gradient(z,[X, Y])
print(v)

#5.....**********************MEAN ABSOLUTE ERROR****************************
#*****************Error = yh-y**********************
a = np.array([2,4,6,8])
ahat= a*3

print(np.mean(np.abs(a-ahat) ))

#5.....**********************MEAN ABSOLUTE ERROR WITH TENSORFLOW****************************
#*****************Error = yh-y**********************
a = tf.Variable([1,2,3,4],dtype=tf.float32)
ahat= tf.Variable(2*a ,dtype=tf.float32)
def mae(a,ahat):
    return tf.reduce_mean(tf.abs(a-ahat))
print(mae(a,ahat))

