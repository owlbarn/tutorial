.PHONY: all
all:
	


.PHONY: clean
clean:
	$(RM) -r all_temp_files


.PHONY: push
push:
	git commit -am "editing book ..." && \
	git push origin `git branch | grep \* | cut -d ' ' -f2`
