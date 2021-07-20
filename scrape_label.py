import re
from urllib.request import urlopen
import pandas
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen


# # # # #

# scrape approach that uses bailii's inbuilt boolean search method as a basis

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import webbrowser
import time
case_links_list = []
client = webbrowser.get('firefox')  # basing initially from https://stackoverflow.com/questions/40161330/how-to-go-to-next-page-using-beautiful-soup
path_to_chromedriver = "/Users/joewatson/Downloads/chromedriver"  # enter path of chromedriver
browser = webdriver.Chrome(executable_path=path_to_chromedriver)
url = "https://www.bailii.org/cgi-bin/lucy_search_1.cgi?highlight=0&sort=date&query=(animal%20OR%20Animal)&datehigh=" \
      "2020&method=boolean&mask_path=uk/cases/UKHL%20uk/cases/UKSC%20uk/cases/UKPC%20ew/cases/EWCA%20ew/cases/EWCA/" \
      "Civ%20ew/cases/EWCA/Crim%20ew/cases/EWHC/Admin%20ew/cases/EWHC/Admlty%20ew/cases/EWHC/Ch%20ew/cases/EWHC" \
      "/Comm%20ew/cases/EWHC/Fam%20ew/cases/EWHC/Mercantile%20ew/cases/EWHC/Patents%20ew/cases/EWHC/QB%20ew/cases" \
      "/EWHC/Costs%20ew/cases/EWHC/TCC&datelow=2000"
browser.get(url)
i = 1
with open("/Users/joewatson/Desktop/LawTech/animal_bailii_bool.txt", "w") as file:
    while True:
        time.sleep(1.7)  # wait some seconds for page loading
        print("end sleep")
        new_soup = BeautifulSoup(browser.page_source, 'html.parser')
        case_links = new_soup.findAll("li")
        for c in case_links:
            url_link = "https://www.bailii.org" + c.contents[0].contents[1].attrs.get('href')
            if url_link not in case_links_list:
                case_links_list.append(url_link)  # extract links
                req = Request(url_link, headers={'User-Agent': 'Mozilla/5.0'})
                webpage = urlopen(req).read()
                page_soup = BeautifulSoup(webpage, "html5lib")
                case_text = page_soup.get_text()
                if "animal" in case_text or "Animal" in case_text:  # https://stackoverflow.com/questions
                    # /21344842/if-a-or-b-in-l-where-l-is-a-list-python
                    animal = "animal_mention"
                else:
                    animal = "no_mention"
                file.write(url_link+ "\t" + animal + "\n")
            print(str(i) + "\t" + url_link)
            i += 1
        if 'Next 10' not in str(new_soup):
            break
        # click next page:
        browser.find_element_by_xpath("//input[@type='submit' and @value='Next 10 >>>']").click()  # solution via
        # https://stackoverflow.com/questions/35531069/find-submit-button-in-selenium-without-id  # Note that in my
        # case the action button is always labelled as 'Next 10' even if less than 10 results left to display

# The above script gens output that shows 1809 cases of which 1800 are unique and of which 1568 are unique + mention 'animal'.
# For this search, bailii claimed to have found 1810 unique cases mentioning 'animal'.

# # # # #

# scrape approach that addresses bailii's inbuilt boolean search tool problems

def bailii_scraper(txt_file, year_min, year_max, court_code):
    with open(txt_file, "w") as file:
        b = 1
        case_years = []
        case_link_list = []
        for i in range(year_min, year_max + 1):
            case_years.append(
                "https://www.bailii.org/" + court_code.lower()[:2] + "/cases/" + court_code + "/" + str(i) + "/")
            for cy in case_years:
                req = Request(cy, headers={'User-Agent': 'Mozilla/5.0'})
                webpage = urlopen(req).read()
                page_soup = BeautifulSoup(webpage, "html5lib")  # html.parser misses a v small proportion of judgments
                case_links = page_soup.findAll("ul")
                for c_links in case_links:
                    clli = c_links.findAll("li")
                    clli = list(set(clli))
                    for c in clli:
                        if str("https://www.bailii.org" + str(c.contents[0].attrs.get('href'))).replace(
                                "cgi-bin/redirect.cgi?path=/", "") \
                                not in case_link_list:
                            # 'not in' to account for some duplicate case links
                            url_link = str("https://www.bailii.org" + str(c.contents[0].attrs.get('href'))).replace(
                                "cgi-bin/redirect.cgi?path=/", "")
                            # .replace method to account for 5s 'official shorthand' message from EWHC/QB/2007/2856 link
                            case_link_list.append(url_link)
                            case_title = c.get_text()  # full case title including date
                            if "[" in case_title and "]" in case_title:
                                case_year = re.findall('\[(.*?)\]', c.get_text())[
                                    -1]  # try [0] for [-1] if errors/anomalies
                            else:
                                case_year = str(i)
                            if "�" in url_link:  # to account for erroneous/empty judgment from EWHC/QB/2007/3358 link
                                animal = "no_judgment_on_bailii"
                            else:
                                req = Request(url_link, headers={'User-Agent': 'Mozilla/5.0'})
                                webpage = urlopen(req).read()
                                page_soup = BeautifulSoup(webpage, "html5lib")
                                case_text = page_soup.get_text()
                                if "animal" in case_text or "Animal" in case_text:  # https://stackoverflow.com/questions
                                    # /21344842/if-a-or-b-in-l-where-l-is-a-list-python
                                    animal = "animal_mention"
                                else:
                                    animal = "no_mention"
                            file.write(court_code + str(b) + "\tCase: " + case_title + "\tYear: " + case_year
                                       + "\tLink: " + url_link + "\tAnimal: " + animal + "\n")
                            b += 1


# to do scrape one court at a time, run:
# bailii_scraper("/Users/joewatson/Desktop/LawTech/EWHC_Mercantile.txt", 2000, 2020, "EWHC/Mercantile")

# to scrape all higher courts, loop over courts_list (below) to create the appropriate combos of function args 1 to 4
courts_list = ["UKPC", "UKHL", "UKSC", "EWCA/Civ", "EWCA/Crim", "EWHC/Admin", "EWHC/Admlty", "EWHC/Ch", "EWHC/Comm",
               "EWHC/Costs", "EWHC/Fam", "EWHC/Mercantile", "EWHC/Patents", "EWHC/QB", "EWHC/TCC"]

for cl in courts_list:
    save_loc = "/Users/joewatson/Desktop/LawTech/" + cl.replace("/", "_") + ".txt"
    if cl == "UKHL":
        start_year = 2000
        end_year = 2009
    elif cl == "UKSC":
        start_year = 2009
        end_year = 2020
    else:
        start_year = 2000
        end_year = 2020
    bailii_scraper(save_loc, start_year, end_year, cl)
    print("done: " + cl)

print("done: All")


tdict = {}
for cl in courts_list:
    tdict[cl.replace("/", "_")] = "/Users/joewatson/Desktop/LawTech/" + cl.replace("/", "_") + ".txt"

lines = 0
tlist = list(tdict.keys())  # needs to be run here (just before loop below) as gets turned into TextWrapper when looping
with open(tdict[tlist[0]]) as tlist[0], open(tdict[tlist[1]]) as tlist[1], open(tdict[tlist[2]]) as tlist[2], \
        open(tdict[tlist[3]]) as tlist[3], open(tdict[tlist[4]]) as tlist[4], open(tdict[tlist[5]]) as tlist[5], \
        open(tdict[tlist[6]]) as tlist[6], open(tdict[tlist[7]]) as tlist[7], open(tdict[tlist[8]]) as tlist[8], \
        open(tdict[tlist[9]]) as tlist[9], open(tdict[tlist[10]]) as tlist[10], open(tdict[tlist[11]]) as tlist[11], \
        open(tdict[tlist[12]]) as tlist[12], open(tdict[tlist[13]]) as tlist[13], open(tdict[tlist[14]]) as tlist[14], \
        open("/Users/joewatson/Desktop/LawTech/animal.txt", "w") as file_animals:
    i = 1
    for tl in tlist:
        for line in tl:
            lines += 1
            if "animal_mention" in line:
                file_animals.write("Line: " + str(i) + "\tPersonal_code: " + line)
                i += 1

print("Number of judgments checked for the presence of 'animal' or 'Animal': " + str(lines))
# 55,202
print("Number of judgments found to contain 'animal' or 'Animal': " + str(i-1))
# 1,637
print("Percentage of judgments featuring 'animal' or 'Animal': " + str(round(100 * (i-1) / lines, 2)))
# 2.97% (to 2dp)


# # # CREATE CSV FILE FROM TXT FILE # # #

names_list = ["Line", "Personal_code", "Case", "Year", "Link", "Animal"]
animal_df = pandas.read_csv("/Users/joewatson/Desktop/LawTech/animal.txt", delimiter="\t", names=names_list)
for nl in names_list:
    animal_df[nl] = animal_df[nl].str.replace(str(nl + ": "), "")  # remove the word descriptor plus colon before info
animal_df['Index'] = range(1, len(animal_df)+1)
animal_df_trim = animal_df[["Index", "Case", "Year", "Link"]]
animal_df_trim['Classification'] = ' '  # copy warning - ignore
animal_df_trim['Explanation'] = ' '  # copy warning - ignore
animal_df_trim_sample = list(animal_df_trim['Index'].sample(n=200, random_state=1))  # select 200 cases for labelling (soon increased to 500 total)
animal_df_trim["Sample"] = np.where(animal_df_trim["Index"].isin(animal_df_trim_sample), 1, 0)  # To label, filter
# by 'Sample' column. This means that the labeller can generally check the whole 'animal'/'Animal' list.
animal_df_trim.to_csv("/Users/joewatson/Desktop/LawTech/animal_df.csv", encoding='utf-8', index=False)

# show cases found that bailii's inbuilt search tool did not find
list_1 = case_links_list
list_2 = list(animal_df_trim["Link"])
main_list = np.setdiff1d(list_2, list_1)

# selecting a further 300 cases for labelling
df = pandas.read_csv("/Users/joewatson/Downloads/Animal law cases for labelling - animal_df.csv")
df2 = df[df['Classification'].isnull()]  # retain non-labelled judgments only
df1 = df[df['Classification'] >= 0]  # retain labelled judgments only
df1['og_sample'] = 1
df2_sample = list(df2['Index'].sample(n=300, random_state=1))
df2["Sample"] = np.where(df2["Index"].isin(df2_sample), 1, 0)  # warning but works fine
df2['og_sample'] = 0
df2 = pd.concat([df1, df2], ignore_index=True)
df2.to_csv("/Users/joewatson/Desktop/LawTech/animal_df2.csv", encoding='utf-8', index=False)

# All judgments containing 'animal' with (and without) human labels can be found in the case_law_repository.csv file in
# the animal_law_classifier GitHub repository