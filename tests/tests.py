from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

opts = Options()
opts.set_headless()
assert opts.headless  # Operating in headless mode

cap = DesiredCapabilities().FIREFOX
cap["marionette"] = False
browser = Firefox(capabilities=cap, executable_path = "/home/geckodriver", options=opts)
browser.get('https://duckduckgo.com')

search_form = browser.find_element_by_id('search_form_input_homepage')
search_form.send_keys('OmicEra')
search_form.submit()

results = browser.find_elements_by_class_name('result')
print(results[0].text)

browser.close()
quit()