
def butter_lowpass(cutoff, nyq_freq, order=6):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=6):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y
# Filter signal x, result stored to y: 
def filter_noise_gravity(x,fc,sample_rate=100):
    """Function to remove gravity comp from readings and
        filtering the "gravity-free" signal
    input:
            x:signal
            fc:cut-off freq filter
    output:
            filtered_signal ,gravity_comp
    """
    cutoff_frequency_gravity = 0.3
    gravity_comp = butter_lowpass_filter(x, cutoff_frequency_gravity, sample_rate/2)
    diff = np.array(x)-np.array(gravity_comp)
    filtered_signal=butter_lowpass_filter(diff,fc,sample_rate/2)
    return filtered_signal,gravity_comp

def filter_noise(x,fc,sample_rate=100):
    """Function to filter high freq noise from readings
    input:
            x:signal
            fc:cut-off freq filter
    output:
            filtered_signal 
    """
    filtered_signal = butter_lowpass_filter(x, fc, sample_rate/2)
    return filtered_signal

def components_processing(sensors_data,fc,sample_rate=100):
    for component in sensors_data.columns:
        signal=sensors_data[component].values
        if("Acc" in component):
            filtered_signal,gravity_comp=filter_noise_gravity(signal,fc,sample_rate)
            #get gravity_free filter
            sensors_data[component ]=filtered_signal
            #total acc
            # sensors_data["{}_total".format(component)]=filtered_signal+gravity_comp
        elif("Giro" in component):
            sensors_data[component]=filter_noise(signal,fc,sample_rate)
    return sensors_data

##EDA 
def univariate_outlier(df,feat,outliers_fraction=0.05):
    X = df[feat].values.reshape(-1,1)
    # Isolation Forest classifier
    clf= IForest(contamination=outliers_fraction)
    clf.fit(X)
    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)
    return y_pred.tolist()

def get_outliers_dataset(dataset,components,features,outliers_fraction=0.05):
    # components=["AccX","AccY","AccZ"]
    # features=["mean","standard_deviation"]
    outlier_dict={}
    for component in tqdm(components):
        for feat in features:
            feat="{0}__{1}".format(component,feat)
            outlier_dict[feat]=univariate_outlier(dataset,feat,outliers_fraction)
    
    outlier_list=np.any(np.stack(list(outlier_dict.values()),axis=1),axis=1)

    return outlier_dict,outlier_list