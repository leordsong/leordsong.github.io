---
title: "{{ replace .File.ContentBaseName "-" " " | title }}"
date: {{ .Date }}
year: {{ now.Year }}
draft: true
---
