#!/usr/bin/env python
# coding: utf-8

# In[94]:


from pytuning.scales import *
import sympy as sp
import numpy as np

# Equal Temperament

def create_edo_template(ref_pitch):
    edo_scale = create_edo_scale(12)
    edo_scale_float = [i.evalf() for i in edo_scale]

    return ref_pitch * np.array(edo_scale_float)

# Just

def create_just_template(ref_pitch):
    just_scale = [sp.Integer(1), sp.Rational(25, 24), sp.Rational(9,8), sp.Rational(6,5), 
                  sp.Rational(5,4), sp.Rational(4, 3), sp.Rational(45, 32), sp.Rational(3,2), 
                  sp.Rational(8,5), sp.Rational(5,3), sp.Rational(9,5), sp.Rational(15,8),
                                                                                 sp.Integer(2)]
    just_scale_float = [i.evalf() for i in just_scale]

    return ref_pitch * np.array(just_scale_float)

# Pythagorean

def create_pythagorean_template(ref_pitch):
    pythagorean_scale = create_pythagorean_scale()
    pythagorean_scale_float = [i.evalf() for i in pythagorean_scale]

    return ref_pitch * np.array(pythagorean_scale_float)

# Template Creation
def create_scale_template(ref_pitch, tuning_system):
    '''
    Creates a template for given scale based on reference pitch. 
    Scale: 'just', 'edo', 'pythagorean'
    '''
    if tuning_system == 'just':
        return create_just_template(ref_pitch)
    if tuning_system == 'edo':
        return create_edo_template(ref_pitch)
    if tuning_system == 'pythagorean':
        return create_pythagorean_template(ref_pitch)
    
    else:
        return 'unrecognized scale'






