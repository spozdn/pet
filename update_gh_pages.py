import os

os.system("rm -r ../build/*")
os.chdir("./docs")
os.system("make html")
os.chdir("../")
os.system("git checkout -f gh-pages")
os.system("git rm -r *")
os.system("cp -r ../build/html/* .")
with open(".nojekyll", "w") as f:
    pass

os.system("git add *")
os.system("git add .nojekyll")
os.system("git commit -m 'automatic docs build'")
os.system("git push")
os.system("git checkout main")
