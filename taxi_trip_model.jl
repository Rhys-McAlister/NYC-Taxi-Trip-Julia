using DataFrames
using CSV
using Parquet2
using Statistics
using Dates
using MLJ

function import_data(file::String)
    data = Parquet2.Dataset(file)
    df = DataFrame(data; copycols=false)
    return df
end

yellow_2022_01 = import_data("yellow_tripdata_2022-01.parquet")

function impute_median(df::DataFrame)
    median_cols = [:passenger_count, :airport_fee, :congestion_surcharge, :RatecodeID]
    for col in median_cols
        df[!, col] = coalesce.(df[!, col], median(skipmissing(df[!, col])))
    end
end

# function impute_median(df::DataFrame)
#     median_cols = [:passenger_count, :airport_fee, :congestion_surcharge, :RatecodeID]
#     median_passenger_count = median(skipmissing(df[!, :passenger_count]))
#     for col in median_cols
#         df[!, col] = replace.(df[!, col], missing => ] )
#     end
# end

impute_median(yellow_2022_01)



function create_date_features(df::DataFrame)
    df.pickup_year = year.(df.tpep_pickup_datetime)
    df.pickup_month = month.(df.tpep_pickup_datetime)
    df.pickup_day = day.(df.tpep_pickup_datetime)
    df.pickup_hour = hour.(df.tpep_pickup_datetime)
    df.pickup_minute = minute.(df.tpep_pickup_datetime)
    df.pickup_second = second.(df.tpep_pickup_datetime)
    df.pickup_weekday = dayname.(df.tpep_pickup_datetime)
    df.pickup_week = week.(df.tpep_pickup_datetime)
end 


create_date_features(yellow_2022_01)

function drop_cols(df::DataFrame)
    drop_cols = [:VendorID, :tpep_pickup_datetime, :tpep_dropoff_datetime]
    for col in drop_cols
        select!(df, Not(col))
    end
end

drop_cols(yellow_2022_01)

describe(yellow_2022_01)


# TODO: one-hot encode vendorid and store_and_fwd_flag
function preprocess(df::DataFrame, predicitors::Vector)
    processed_df = df[:, predicitors]
    processed_df = coerce!(processed_df, Union{Missing, Continuous}=>Continuous, tight=true)
    return processed_df
end

function train_test_split(df::DataFrame)
    

    y, X = MLJ.unpack(processed_df, ==(Symbol("trip_distance")), colname -> true)
    train, test = partition(eachindex(y), 0.7)
    return y, X, train, test
end



function linear_regression_model()
    OLS = @load LinearRegressor pkg=GLM
    ols = OLS()
    mach =  machine(ols, X, y) |> fit!
    fitted_param = fitted_params(mach)
    reports = report(mach)
    return fitted_param, reports
end


processed_df = preprocess(yellow_2022_01, [:trip_distance, :total_amount, :passenger_count])

y, X, train,test = train_test_split(processed_df)

fitted_param, reports = linear_regression_model()