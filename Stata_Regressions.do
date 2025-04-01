cap cd "C:\Users\jcabr\OneDrive - University of Toronto\Coursework\Winter\ECO2460\Empirical Project\Data"

***************************
* Processing the EPU Data *
***************************
*Downloaded from: https://www.policyuncertainty.com/global_monthly.html

*Canadian EPU Index
import excel "Canada_Policy_Uncertainty_Data.xlsx", clear firstrow
destring Year, replace
rename CanadaNewsBasedPolicyUncerta EPU_Canada
save "EPU_Canada.dta", replace

*Global EPU Index
import excel "Global_Policy_Uncertainty_Data.xlsx", clear firstrow
destring Year, replace
rename GEPU_current EPU_Global
save "EPU_Global.dta", replace

*American daily EPU Index
import excel "USEPUINDXD.xlsx", sheet("Daily, 7-Day") clear firstrow
gen Year = year(observation_date)
gen Month = month(observation_date)
gen Day = day(observation_date)
drop observation_date 
rename USEPUINDXD EPU_USA
save "EPU_USA.dta", replace

***************************
* Processing the VIX Data *
***************************

*Canadian TSX Volatility
import excel "TSX Volatility 2015-2025", clear firstrow
rename SPTSX60Index SPTSX
gen Year = year(Effectivedate)
gen Month = month(Effectivedate)
gen Day = day(Effectivedate)
drop Effectivedate
save "SPTSX.dta", replace

import delimited "VIX S&P Volatility Index 1990-2025", clear
rename vixcls vix
gen datevar = date(observation_date, "YMD")
format datevar %td
gen Year = year(datevar)
gen Month = month(datevar)
gen Day = day(datevar)
drop observation_date
save "VIX.dta", replace

**************************************
* Generate some additional variables *
**************************************

import delimited "Final_Data_Reg.csv", clear

*remove the topic that will be the "base case". Choosing topic 4 which seems to be regular hearing jargon (colleague, issue, committee, legislation, ...)
*note: topic codes are offset by 1 from python indexing
drop topic_3


gen datevar = date(date, "YMD")
format datevar %td
gen Year = year(datevar)
gen Month = month(datevar)
gen Day = day(datevar)


*merge the EPU data
merge m:1 Month Year using "EPU_Canada.dta", keep(1 3) nogen
merge m:1 Month Year using "EPU_Global.dta", keep(1 3) nogen
merge m:1 Month Year Day using "EPU_USA.dta", keep(1 3) nogen

*merge the stock market indices
merge m:1 Month Year Day using "SPTSX.dta", keep(1 3) nogen
merge m:1 Month Year Day using "VIX.dta", keep(1 3) nogen

*encode some variables to use as FE

encode last_gender, gen(last_gender_code)
encode current_speaker, gen(speaker_code)
encode party, gen(party_code)
encode date, gen(day_code)

gen female = (gender == "F")
gen last_female = (last_gender == "F")

*last party codes (last party being "none" is the base case)
gen last_party_bq = ((last_party == "BQ") & party != "BQ")
gen last_party_con = ((last_party == "CPC") & party != "CPC")
gen last_party_lib = ((last_party == "Lib.") & party != "Lib.")
gen last_party_ndp = ((last_party == "NDP") & party != "NDP")
gen last_party_gp = ((last_party == "GP") & party != "GP")

*inpower
gen lib_power = (datevar >= date("2015-11-04", "YMD"))
gen inpower = ((party == "Lib.") & (datevar >= date("2015-11-04", "YMD"))) | ((party == "CPC") & (datevar < date("2015-11-04", "YMD")))


*standardize the dependent variable, stock market indices and EPU
foreach ind in sent_hostile EPU_Canada EPU_Global EPU_USA vix SPTSX {
	egen mean_`ind' = mean(`ind')
	egen sd_`ind' = sd(`ind')
	gen `ind'_std = (`ind' - mean_`ind') / sd_`ind'
	
}

*generate the interaction terms
gen Other_EPU_Can = EPU_Canada_std*mention_any
gen Other_EPU_Global = EPU_Global_std*mention_any
gen Other_EPU_USA = EPU_USA_std*mention_any

gen Other_SPTSX = SPTSX_std*mention_any
gen Other_VIX = vix_std*mention_any

*log the dependent variable for interpretability
*adjustment for zero values
*replace sent_hostile = 0.00000000001 if sent_hostile == 0
*gen ln_hostile = log(sent_hostile)


*****************************
* Run the Final Regressions *
*****************************
gen yq = yq(year(datevar), quarter(datevar))  
format yq %tq
egen partyyear = group(party lib_power)


local controls original_length female lib_power interrupted inpower last_party_*


*to get a consistent sample
*some (less than 10) observations get dropped when we include speaker FE (probably speakers who only speak once in the whole sample)
*reghdfe sent_hostile_std Other_EPU_Can Other_EPU_Global EPU_Canada_std EPU_Global_std mention_any `controls' topic_*, absorb(speaker_code party_code Year)
*gen insample = e(sample)

* (1) year FE, controls
reghdfe sent_hostile_std Other_EPU_Can Other_EPU_Global EPU_Canada_std EPU_Global_std mention_any `controls' if insample == 1, absorb(Year)

est store T1_C1
estadd local topic "No", replace
estadd local controls "No", replace
estadd local speakerfe "No", replace

* (2) year, topic
reghdfe sent_hostile_std Other_EPU_Can Other_EPU_Global EPU_Canada_std EPU_Global_std mention_any `controls' topic_* if insample == 1, absorb(Year)

est store T1_C2
estadd local topic "Yes", replace
estadd local controls "No", replace
estadd local speakerfe "No", replace

* (3) year, party FE, controls, Topics
reghdfe sent_hostile_std Other_EPU_Can Other_EPU_Global EPU_Canada_std EPU_Global_std mention_any `controls' topic_* if insample == 1, absorb(party_code Year)

est store T1_C3
estadd local topic "Yes", replace
estadd local controls "Yes", replace
estadd local speakerfe "No", replace

* (4) year, party, speaker FE, controls, Topics
reghdfe sent_hostile_std Other_EPU_Can Other_EPU_Global EPU_Canada_std EPU_Global_std mention_any `controls' topic_* if insample == 1, absorb(speaker_code party_code Year)

est store T1_C4
estadd local topic "Yes", replace
estadd local controls "Yes", replace
estadd local speakerfe "Yes", replace


esttab T1_C1 T1_C2 T1_C3 T1_C4 using "Main_Table.tex", replace ///
b(3) se(3) compress nomtitles booktabs fragment noconstant nonotes gaps ///
order(Other_EPU_Can Other_EPU_Global EPU_Canada_std EPU_Global_std ///
      mention_any female inpower lib_power interrupted) ///
keep(Other_EPU_Can Other_EPU_Global EPU_Canada_std EPU_Global_std ///
     mention_any female inpower lib_power interrupted) ///
stats(topic controls speakerfe N r2, fmt(0 0 0 0 0) layout(@ @ @ @ @) ///
labels("LDA Topics" "Party FE" "Speaker FE" "Observations" "R^{2}")) ///
starlevels(* 0.10 ** 0.05 *** 0.01) brackets nolines ///
posthead("\midrule") postfoot("\midrule")
estimates clear 

STOP

***********************************************
* Robustness Test: Daily Stock Market Indices *
***********************************************
*generate year-month FE
gen ym = ym(year(datevar), month(datevar))  
format ym %tm

local controls original_length female lib_power interrupted inpower last_party_*


*get a consistent sample
cap drop insample
reghdfe sent_hostile_std Other_SPTSX Other_VIX SPTSX_std vix_std mention_any `controls' topic_*, absorb(speaker_code party_code ym)
gen insample = e(sample)

* (1) No FE, controls
reghdfe sent_hostile_std Other_SPTSX Other_VIX SPTSX_std vix_std mention_any `controls' if insample == 1, absorb(ym)
est store T2_C1
estadd local topic "No", replace
estadd local controls "No", replace
estadd local speakerfe "No", replace

* (2) year, topic
reghdfe sent_hostile_std Other_SPTSX Other_VIX SPTSX_std vix_std mention_any `controls' topic_* if insample == 1, absorb(ym)
est store T2_C2
estadd local topic "Yes", replace
estadd local controls "No", replace
estadd local speakerfe "No", replace

* (3) year, party FE, controls, Topics
reghdfe sent_hostile_std Other_SPTSX Other_VIX SPTSX_std vix_std mention_any `controls' topic_* if insample == 1, absorb(party_code ym)
est store T2_C3
estadd local topic "Yes", replace
estadd local controls "Yes", replace
estadd local speakerfe "No", replace

* (4) year, party, speaker FE, controls, Topics
reghdfe sent_hostile_std Other_SPTSX Other_VIX SPTSX_std vix_std mention_any `controls' topic_* if insample == 1, absorb(speaker_code party_code ym)
est store T2_C4
estadd local topic "Yes", replace
estadd local controls "Yes", replace
estadd local speakerfe "Yes", replace

esttab T2_C1 T2_C2 T2_C3 T2_C4 using "Main_Table_Stocks.tex", replace ///
b(3) se(3) compress nomtitles booktabs fragment noconstant nonotes gaps ///
order(Other_SPTSX Other_VIX SPTSX_std vix_std ///
      mention_any female lib_power inpower) ///
keep(Other_SPTSX Other_VIX SPTSX_std vix_std ///
     mention_any female lib_power inpower) ///
stats(topic controls speakerfe N, fmt(0 0 0 0) layout(@ @ @ @) ///
labels("LDA Topics" "Party FE" "Speaker FE" "Observations")) ///
starlevels(* 0.10 ** 0.05 *** 0.01) brackets nolines ///
posthead("\midrule") postfoot("\midrule")
estimates clear 
