This repository just contains random notes.
To get a nicer html5 file including rendered LaTeX formulas execute:

        x="<your-md-file>"
        pandoc -s --toc --toc-depth=2 -t html5 -c pandoc.css -H header.html "$x" LaTeXmetadata.yaml -o ${x/%md/}html
