#!/usr/bin/env python
# coding: utf-8

# In[1]:


from argparse import ArgumentParser


parser = ArgumentParser()

# Add flags
parser.add_argument('--cities', nargs="+", default=["la", "paris"])
parser.add_argument('--styles', nargs="+", default=["CartoVoyagerNoLabels", "StamenTonerBackground"])
parser.add_argument('--zooms', nargs="+", default=["14"])

# Parse
args, unknown = parser.parse_known_args()
print(args)
print(unknown)
