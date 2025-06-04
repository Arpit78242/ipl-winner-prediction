import pickle
import streamlit as st
import pandas as pd

st.title("IPL Winner predictor in 2nd Innings")

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bengaluru',
 'Kolkata Knight Riders',
 'Punjab Kings',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals',
 'Gujarat Titans']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open("pipe.pkl","rb"))

col1, col2, col3 = st.columns(3)

with col1:
    batting_team = st.selectbox("Select the Batting team:", sorted(teams))

with col2:
    bowling_team = st.selectbox("Select the Bowling team:", sorted(teams), index = 4)

with col3:
    city = st.selectbox("Select the organizing city: ", sorted(cities), index = 8)

col4, col5 = st.columns(2)

with col4:
    target = st.number_input("Targeted Score: ", min_value = 0, step = 1, format = "%d")

with col5:
    runs = st.number_input("Current Score: ", min_value = 0, step = 1, format = "%d")

col6, col7, col8 = st.columns(3)

with col6:
    over = st.number_input('Overs completed', min_value=0, max_value=20, step=1, format="%d")
with col7:
    balls = st.number_input("Balls in current over: ", min_value = 1, max_value = 6, step = 1, format = "%d")
with col8:
    wickets = st.number_input("Wickets: ", min_value = 0, max_value = 9, step = 1, format = "%d")


if st.button("Predict Probability"):
    runs_left = target - runs
    balls_left = 120 - (over*6 + balls)
    wickets_left = 10 - wickets
    crr = (runs*6) / (6*over + balls)
    rrr = (runs*6) / (balls_left + 0.0001)
    rw_ratio = runs/(wickets + 0.0001)

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets_left],'total_runs_x':[target],'crr':[crr],'rrr':[rrr],'r/w_ratio':[rw_ratio]})

    result = pipe.predict_proba(input_df)

    loss = result[0][0]
    win = result[0][1]

    # st.write(input_df)

    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")