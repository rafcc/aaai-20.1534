SIZE := 10000
RHO := 1e-5
EPS := 1e-4
SEED := 42

EN := src/elastic_net.py --size $(SIZE) --eps $(EPS) --seed $(SEED)
GL := src/group_lasso.py --size $(SIZE) --rho $(RHO) --eps $(EPS) --seed $(SEED)
DATA_DIR := data
DATASETS := $(wildcard $(DATA_DIR)/*.csv)
SUFFIXES_W := ,w_1 ,w_2 ,w_3 ,w_1_2 ,w_1_3 ,w_2_3 ,w_1_2_3
SUFFIXES_X := $(subst w,x,$(SUFFIXES_W))
SUFFIXES_F := $(subst w,f,$(SUFFIXES_W))
SUBSAMPLES_W := $(foreach s,$(SUFFIXES_W),$(addsuffix $(s),$(DATASETS)))
SUBSAMPLES_X := $(foreach s,$(SUFFIXES_X),$(addsuffix $(s),$(DATASETS)))
SUBSAMPLES_F := $(foreach s,$(SUFFIXES_F),$(addsuffix $(s),$(DATASETS)))

group-lasso: data/Birthwt.csv data/Birthwt_group.csv
	$(GL) $^ 1 2 3

elastic-net: $(SUBSAMPLES_W)

clean:
	$(RM) $(SUBSAMPLES_W) $(SUBSAMPLES_X) $(SUBSAMPLES_F)

%.csv,w_1: %.csv
	$(EN) $< 1
%.csv,w_2: %.csv
	$(EN) $< 2
%.csv,w_3: %.csv
	$(EN) $< 3

%.csv,w_1_2: %.csv
	$(EN) $< 1 2
%.csv,w_1_3: %.csv
	$(EN) $< 1 3
%.csv,w_2_3: %.csv
	$(EN) $< 2 3

%.csv,w_1_2_3: %.csv
	$(EN) $< 1 2 3

