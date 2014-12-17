function [X,y]  = parse_data(fname, num_labels)

[dataName,attributeName, attributeType, data] = arffread(fname);
num_labels
size(data)
X = data(:,1:end-(num_labels+2));
y = data(:,end-num_labels+1:end);

end
