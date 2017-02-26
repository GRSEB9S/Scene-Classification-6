function [ label_names ] = load_label_names( label_file )
    [name_list, labels] = textread(label_file, '%s %d');
    for i = 1:length(name_list)
        name=name_list{i};
        name=string_split(name);
        name_list{i}=name{3};
    end
    label_names=containers.Map(labels,name_list);
end

