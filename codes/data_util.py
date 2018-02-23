"""

Utility functions for getting datasets.
In particular:
(1) Macaulay Library of Cornell Lab of Ornithology
(2) Xeno-Canto Database

"""

import urllib2
import re
from bs4 import BeautifulSoup as Soup
import subprocess
import glob
import numpy as np
import pandas as pd
from collections import OrderedDict

"""
 ----------Macaulay Library----------
"""

class ML_dn(object):
    def __init__(self,preurl,num_spe=100,ass_preurl='https://macaulaylibrary.org/asset/',read_in=True,pglim=None):
        """
        Inputs
        - preurl : 
            string of url (without page num) from the search result of ML library: 
             http://macaulaylibrary.org/search?media_collection=1
             
        - num_spe :
            Number of species to download data. Ordered by frequency of occurrence
            in the ML database (descending). Defaults to 100.
        - ass_preurl : 
            string of url of the individual ML catalog asset prefix. Defaults to
             https://macaulaylibrary.org/asset/
        
        """
        self.preurl = preurl
        self.num_spe = num_spe
        self.ass_preurl = ass_preurl
        
        ## read in all pages
        if read_in:
            if pglim:
                self.url, self.content = self._url_pages(self.preurl,pglim)
            else:
                self.url, self.content = self._url_pages(self.preurl)
        
        self.spenmlist = [] # initialize
        
    def find_MLnum(self):
        """
        Retrieve ML catalog number of this search from url list
        """
        MLnumlist = []
        for i in range(len(self.url)):
            s = Soup(self.content[i], 'html.parser')
            audio = s.findAll(href=re.compile("audio/"))
            
            for node in audio:
                MLnumlist.append(re.sub(r'\D|\n', '', node.text))
                MLnumlist[:] = [int(nm) for nm in MLnumlist if nm != '']
                # remove whitespaces and empty lines and convert to integers
                
        ## remove possible duplicates
        self.MLnumlist = list(set(MLnumlist))      
        
        return self.MLnumlist
    
    def get_speName(self):
        """
        Retrieve taxonomic/species name of the species from ML number
        """
        
        for i,cnt in enumerate(self.content):
            s = Soup(cnt,'html.parser') 
            ss = s.findAll(class_='indent')
            
            for j in range(0,len(ss),3):
                try:
                    name = re.search("\n(.+?)\n<div>", str(ss[j])).group(1)
                    tmp  = re.sub("[\(\[].*?[\)\]]", "", name).rstrip().lstrip()
                    
                    self.spenmlist.append(tmp.replace(' ','%20').replace("'",'%27'))
                    
                except:
                    pass
        
        return self.spenmlist
    
    def get_specsv(self):
        """
        Download csv for a given search result url.
        Default: https://search.macaulaylibrary.org/catalog?view=Grid&req=true&mediaType=a&q=<species>
        """
        from selenium import webdriver
        from selenium.webdriver.firefox.firefox_profile import FirefoxProfile

        profile = FirefoxProfile('/Users/jasonhc/Library/Application Support/Firefox/Profiles/q20kmjze.default')
        profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream, text/csv")
        driver = webdriver.Firefox(firefox_profile=profile)
        
        _ = self._spenm_one()
        
        for i,nm in enumerate(self.spenmlist_one):
            if i > self.num_spe:
                break
            driver.get('https://search.macaulaylibrary.org/catalog?view=Grid&req=true&mediaType=a&q='+nm)
            driver.find_element_by_link_text('Save Spreadsheet').click()
            
    #--- Download mp3
    def dn_mp3(self,path_csv='.',path_mp3='.'):
        """
        Download mp3 according to the csv file (catalog number)
        Can be run independently once the csv's are acquired
        
        Inputs
        - path : 
            The paths of the csv/mp3 files. Default to current working directory
        """
        
        for file_name in glob.glob(path_csv+'*.csv'):
            f = pd.read_csv(file_name,delimiter=',')
    
            catalog = np.asarray(f['ML Catalog #'])
            comname = np.asarray(f['Common Name'])
            comname = [re.sub("[\(\[].*?[\)\]]", "", name).rstrip().lstrip().\
                       replace(' ','').replace("'","").replace('/','-') for name in comname]
            
            sinname = list(set(comname))            
            
            subprocess.call(['cd '+path_mp3],shell=True)
            for sp in sinname:
                subprocess.call(['mkdir '+path_mp3+'/'+sp],shell=True)
    
            for i,cat in enumerate(catalog):
                subprocess.call(['curl -s https://download.ams.birds.cornell.edu/api/v1/asset/'+str(cat)+ \
                                 ' > '+str(comname[i])+str(i+1).zfill(3)+'.mp3'],shell=True)
                subprocess.call(['mv '+str(comname[i])+str(i+1).zfill(3)+'.mp3 '+path_mp3+'/'+str(comname[i])+'/'],shell=True)
            
    ##############################   
    def _url_pages(self,preurl,pglim=10000):
        """
        Create list of all pages from a ML library archival search pre-url
        """
        pg = 1
        urllist, content = [],[]
        for i in range(pglim):
            if not (pg%10): print "Current page: %d" %pg
            try:
                urllist.append(preurl+'&page='+str(pg))
                content.append(urllib2.urlopen(urllist[pg-1]).read())
                pg += 1
            except: # if no more page
                urllist = urllist[:-1] # remove extra entry
                break
                
        return urllist, content
    
    def _spenm_one(self):
        """
        Return species names from the search result, ordered by occurrences
        """
        ## Obtain ML catalog names if haven't done so
        if not hasattr(self,'spenmlist'):
            _ = self.get_speName()
        
        cp = self.spenmlist[:]               ## make copies
        self.spenmlist_one = self.spenmlist[:]

        ## sort decending according to freq of occurrence
        self.spenmlist_one.sort(key=lambda x: cp.count(x), reverse=True)
        self.spenmlist_one = list(OrderedDict.fromkeys(self.spenmlist_one))
        
        return self.spenmlist_one
    
    #--------------------------#
    def get_speName_old(self):
        """
        (Obsolete)
        Retrieve taxonomic/species name of the species from ML number
        (Can actually not do find_MLnum. Original thoughts from there did not work)
        """
        ## Obtain ML catalog numbers if haven't done so
        if not hasattr(self,'MLnumlist'):
            _ = self.find_MLnum()
        
        spenmlist = []
        for i,num in enumerate(self.MLnumlist):
            t = urllib2.urlopen('https://macaulaylibrary.org/asset/'+str(num)).read()
            s = Soup(t,'html.parser') 
            
            spenmlist.append((str(s.title.string).replace('Macaulay Library','')\
                                                 .replace('ML'+str(num),'')).lstrip().rstrip()\
                              .replace("'",'%27'))
            spenmlist[i] = re.sub("[\(\[].*?[\)\]]", "", spenmlist[i]).rstrip().replace(' ','%20')
            ## remove () []
            
            
        self.spenmlist = spenmlist
        return self.spenmlist




"""
----------Xeno-Canto----------
"""
class XC_dn(object):
    def __init__(self,preurl,num_spe=100,speurl='https://www.xeno-canto.org/explore?query=',read_in=True,pglim=None):
        """
        Inputs
        - preurl : 
            string of url (without page num) from the search result of Xeno-Canto database: 
             https://www.xeno-canto.org/explore?query=<species>
             
        - num_spe :
            Number of species to download data. Ordered by frequency of occurrence
            in the database (descending). Defaults to 100.
            
        - speurl :
            string of url for individual species search page
            Defaults to 'https://www.xeno-canto.org/explore?query=<species>'
        
        """
        self.preurl = preurl
        self.num_spe = num_spe
        self.speurl = speurl
        
        ## read in all pages
        if read_in:
            if pglim:
                self.url, self.content = self._url_pages(self.preurl,pglim)
            else:
                self.url, self.content = self._url_pages(self.preurl)
        
    def get_speName(self):
        """
        Retrieve taxonomic/species name of the species from the search
        """
        
        spenmlist = []
        for i,cnt in enumerate(self.content):
            s = Soup(cnt,'html.parser') 
            ss = s.findAll(class_='common-name')
            
            for j in range(len(ss)):
                st = Soup(str(ss[j]), 'html.parser')
                spenmlist.append(st.a.string.replace(' ','+').replace("'",'%27'))
            
        self.spenmlist = spenmlist
        return self.spenmlist
    
    #--- Download mp3
    def dn_mp3(self,num_lim=True,**kwargs):
        """
        Download mp3 according to the species search page
        """
        
        ## create directories for species
        self._make_dir_spe()
        
        for k,spe in enumerate(self.spenmlist_one):
            if num_lim and k > self.num_spe:
                break            
            ## obtain all pages of this species          
            urll, content = self._url_pages(self.speurl+spe,**kwargs)
            
            l = 0
            for i,cnt in enumerate(content):
                s = Soup(cnt,'html.parser')
                st = s.findAll(class_='jp-type-single')
                
                for j in range(len(st)):                    
                    try:                     
                        dnurl = re.search('filepath="//(.+?)mp3"', str(st[j])).group(1)                    
                        ## download                        
                        subprocess.call(['curl -s '+'https://'+dnurl+'mp3'\
                                         ' > '+str(spe.replace('+','').replace("'","%27"))+str(l+1).zfill(3)+'.mp3'],shell=True)
                        subprocess.call(['mv '+str(spe.replace('+','').replace("'","%27"))+str(l+1).zfill(3)+'.mp3 '+\
                                               str(spe.replace('+','').replace("'","%27"))+'/'],shell=True)
                        l += 1
                    except:
                        continue
                
        
    ###################################
    def _url_pages(self,preurl,pglim=10000):
        """
        Create list of all pages from a Xeno-Canto library archival search pre-url
        """
        pg = 1
        urllist, content = [],[]
        for i in range(pglim):
            if not (pg%10): print 'Current page: %d'%pg
            try:
                urllist.append(preurl+'&pg='+str(pg))
                content.append(urllib2.urlopen(urllist[pg-1]).read())
                pg += 1
            except: # if no more page
                urllist = urllist[:-1] # remove extra entry
                break
                
        return urllist, content
    
    def _spenm_one(self):
        """
        Return species names from the search result, ordered by occurrences
        """
        ## Obtain ML catalog names if haven't done so
        if not hasattr(self,'spenmlist'):
            _ = self.get_speName()
        
        cp = self.spenmlist[:]               ## make copies
        self.spenmlist_one = self.spenmlist[:]

        ## sort decending according to freq of occurrence
        self.spenmlist_one.sort(key=lambda x: cp.count(x), reverse=True)
        self.spenmlist_one = list(OrderedDict.fromkeys(self.spenmlist_one))
        
        return self.spenmlist_one
    
    def _make_dir_spe(self,num_lim=True):
        """
        Make directories for species to be downloaded
        """
        for i,spe in enumerate(self.spenmlist_one):
            if num_lim and i > self.num_spe:
                break
            subprocess.call(['mkdir '+spe.replace('+','')],shell=True)