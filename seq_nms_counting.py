from operator import itemgetter
import torch
import cv2 
from seq_nms import seq_nms
from utils.datasets import LoadImages
from pathlib import Path
import numpy as np
from compute_overlap import compute_overlap_areas_given, compute_area
import networkx 
from networkx.algorithms.components.connected import connected_components

# Entry point for the object counting algorithm
def perform_seq_nms(numpy_boxes, numpy_scores, numpy_labels):    
    # If you want to save the results from YOLO for later testing
    # np.save("test_cows_box", numpy_boxes)
    # np.save("test_cows_score", numpy_scores)
    # np.save("test_cows_label", numpy_labels)

    numpy_boxes, numpy_scores, numpy_labels, num_objects, total_object_count, frame_object_count = build_box_sequences(numpy_boxes, numpy_scores, numpy_labels)
    return numpy_boxes, numpy_scores, numpy_labels, num_objects, total_object_count, frame_object_count

# This was modified from YOLO
def annotate_video(source, imgsz, stride, boxes, scores, labels, names, device, save_path, ext="-seq-nms.", total_obj_count=None, frame_obj_count=None):
    print("Annotating Video")
    # Convert the Save path to add -seq-nms to the name
    save_path = save_path.split("/")
    new_ext = save_path[-1].split(".")
    new_ext.insert(-1, ext)
    new_ext = "".join(new_ext)
    save_path[-1] = new_ext
    save_path = "/".join(save_path)
   

    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    i = 0
    vid_path = None

    for path, img, im0s, vid_cap in dataset:
        print("\tannotating frame", i)
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        boxes[i] = scale_coords(img.shape[2:], boxes[i], im0s.shape).round()
        for box, score, label in zip(boxes[i], scores[i], labels[i]):
            # print(label)
            # print(names)
            label_text = f'{names[int(label)]} {score:.2f}'
            plot_one_box(box, im0s, color=[255, 191, 0], label=label_text, line_thickness=3)
        # cv2.imshow(str(Path(path)), im0s)
        # cv2.waitKey(1000)  # 1 millisecond
        if(total_obj_count and frame_obj_count):
            obj_text_1 = "Objects in Frame:" + str(frame_obj_count[i])
            obj_text_2 = "Total Objects:" +  str(total_obj_count[i])
            # cv2.putText(im0s, obj_text, (0,0), "Times")
            cv2.putText(im0s, obj_text_1, (20,100), 0, 2, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(im0s, obj_text_2, (20,200), 0, 2, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)

        # Video Writing
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if vid_path != save_path:  # new video
            vid_path = save_path
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

        vid_writer.write(im0s)

        i+=1

# Taken from YOLO
def plot_one_box(xyxy, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Taken from YOLO
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

# Taken from YOLO
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2

# Converts the YOLO sequence to a padded numpy array suitable for doing SEQ-NMS style algorithms on
def convert_seqence_to_padded_numpy(seq_conf, seq_coords, seq_labels):
    max_len = max([len(x) for x in seq_conf])

    # Pad the boxes so each frame has the same number of confidence boxes
    seq_conf = [torch.tensor(framed_boxes) for framed_boxes in seq_conf]
    padded_seq_conf = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in seq_conf]
    stacked_seq_conf = torch.stack(padded_seq_conf)

    # Pad the labels so each frame has the same number of labels
    seq_labels = [torch.tensor(framed_labels) for framed_labels in seq_labels]
    padded_seq_labels = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in seq_labels]
    stacked_seq_labels = torch.stack(padded_seq_labels)
    
    # Pad the coords so each frame has the same number of coords
    seq_coords = [torch.stack(framed_coords) for framed_coords in seq_coords]
    seq_coords_padded = []
    for frame_coord in seq_coords:
        if len(frame_coord) == max_len: 
            seq_coords_padded.append(frame_coord)
            continue
        padded_tensor = torch.stack([torch.tensor([0,0,0,0]) for x in range(max_len - len(frame_coord))])
        frame_coord = torch.cat((frame_coord, padded_tensor), 0)
        seq_coords_padded.append(frame_coord)

    stacked_seq_coords = torch.stack(seq_coords_padded)

    numpy_boxes = stacked_seq_coords.numpy()
    numpy_scores = stacked_seq_conf.numpy()
    numpy_labels = stacked_seq_labels.numpy()

    return numpy_boxes, numpy_scores, numpy_labels    

# Main entry point for the work done in object counting
def build_box_sequences(boxes, scores, labels, linkage_threshold=0.3, window_size=5, seq_confidence_threshold=0.55):
    ''' Build bounding box sequences across frames. A sequence is a set of boxes that are linked in a video
    where we define a linkage as boxes in adjacent frames (of the same class) with IoU above linkage_threshold (0.5 by default).
    Args
        boxes                  : Tensor of shape (num_frames, num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format. 
        scores                : Tensor of shape (num_frames, num_boxes) containing the confidence score for each box.
        linkage_threshold      : Threshold for the IoU value to determine if two boxes in neighboring frames are linked 
    Returns 
        A list of shape (num_frames - 1, num_boxes, k, 1) where k is the number of edges to boxes in neighboring frame (s.t. 0 <= k <= num_boxes at f+1)
        and last dimension gives the index of that neighboring box. 
    '''

    # Set variables
    linkage_threshold = 0.3
    window_size = 5
    seq_confidence_threshold = 0.5
    small_sequence = 3


    min_print = True
    num_boxes = boxes.shape[1]
    num_frames = boxes.shape[0]
    
    # Find a matrix of overlaps
    overlap_matrix = generate_overlap_matrix(boxes)

    # Find sequences based on overlaps
    sequences = find_sequences_graph_from_overlap(overlap_matrix, num_frames, num_boxes, scores, window_size=window_size, linkage_threshold=linkage_threshold)
    if min_print:
        print("Number of objects BEFORE removing potential false positives", len(sequences))
        print("Potential Sequences found")
        for sq in sequences:
            print("\t", sq) 
    
    start_frames = []
    stop_frames = []
    # Remove any sequences that don't meet thresholds
    sequences = remove_false_positive_sequences(sequences, scores, short_length_threshold=small_sequence, seq_confidence_threshold=seq_confidence_threshold)
    if min_print:
        print("Number of objects AFTER removing potential false positives", len(sequences))
        print("Sequences found")
    for sq in sequences:
        start_frames.append(min(sq, key=lambda x:x[0])[0])
        stop_frames.append(max(sq, key=lambda x:x[0])[0])
        if min_print:
            print("\t", sq) 
    
    # Find missing frames in sequences add add them
    missing_frames_sequences = check_for_missing_links_in_sequences(sequences)
    boxes_to_add = generate_missing_frames(sequences, missing_frames_sequences, boxes, scores, window_size=window_size)
    print(boxes_to_add)
    print("Generated missing frames")
    boxes, scores, labels = create_final_sequences(sequences, boxes, scores, labels, boxes_to_add)
    print("Generated final sequences")
    
    # Get counts for total number of objects so far in the video
    total_object_count = [0 for x in range(num_frames)]
    frame_object_count = [0 for x in range(num_frames)]
    for f_sq in range(len(sequences)):
        start_frame = start_frames[f_sq]
        stop_frame = stop_frames[f_sq]
        total_object_count[start_frame:] = [x + 1 for x in total_object_count[start_frame:]]
        frame_object_count[start_frame:stop_frame + 1] = [x + 1 for x in frame_object_count[start_frame:stop_frame+1]]
        print("\tSequence #", f_sq, "is", start_frame,":", stop_frame) 
    print("Generate object counts")
    print(total_object_count)
    print(frame_object_count)

    return boxes, scores, labels, len(sequences), total_object_count, frame_object_count

# Generates final sequences with missing frames, suitable for annotating. 
def create_final_sequences(sequences, boxes, scores, labels, boxes_to_add):
    number_frames = len(boxes)
    number_sequences = len(sequences)
    new_boxes = np.zeros((number_frames, number_sequences, 4), float)
    new_scores = np.zeros((number_frames, number_sequences), float)
    new_labels = np.zeros((number_frames, number_sequences), int)

    for sqx, seq in enumerate(sequences):
        # print("Sequence #", sqx)      
        # Add existing boxes
        label_ = 0
        for fdx, bdx in seq:
            box_ = boxes[fdx][bdx]
            score_ = scores[fdx][bdx]
            label_ = labels[fdx][bdx]
            new_boxes[fdx][sqx] = box_
            new_scores[fdx][sqx] = score_
            new_labels[fdx][sqx] = label_
        
        # Add the new boxes
        for frame_x, box_x, score_x in boxes_to_add[sqx]:
            new_boxes[frame_x][sqx] = box_x
            new_scores[frame_x][sqx] = score_x
            # use the last known label for the sequence
            new_labels[frame_x][sqx] = labels[fdx][bdx]
    return new_boxes, new_scores, new_labels

# works to remove sequences based on some criteria for False positives.
# Needs tuning based on specific characteristics of video
def remove_false_positive_sequences(sequences, scores, short_length_threshold=2, seq_confidence_threshold=0.5):
    new_sequences = []
    count = 0
    for seq in sequences:

        total_score = 0
        max_score = 0
        for frame, box_index in seq:
            score = scores[frame][box_index]
            total_score+=score
            max_score = max(max_score, score)
        avg_score = total_score / len(seq)

        # Keep sequences with high average score
        if avg_score >= seq_confidence_threshold:
            new_sequences.append(seq)
            continue

        # Keep short sequences with high max scores
        if len(seq) > short_length_threshold and max_score > 0.7:
            new_sequences.append(seq)
            continue
        
        count += 1
        
    print("Removed ", count, "sequences due to size or score thresholds")

    return new_sequences

def generate_missing_frames(sequences: list[set], missing_frames_sequences: list[set], boxes, scores, window_size=5):
    boxes_for_sequences = []
    for sqx, (sequence, missing_frames) in enumerate(zip(sequences, missing_frames_sequences)):
        sequence = sorted(list(sequence), key=itemgetter(0))
        missing_frames = sorted(list(missing_frames))
        print("Sequence #", sqx, sequence)
        if len(missing_frames) == 0:
            boxes_for_sequences.append([])
            print("\tNo missing sequences")
            continue
        
        boxes_to_add = []
        for missing_frame in missing_frames:

            print("\t Looking to add", missing_frame)
            adjacent_boxes = []
            adjacent_scores = 0
            for fdx, bdx in sequence:
                if fdx >= missing_frame - window_size // 2 and fdx <= missing_frame + window_size // 2:
                    adjacent_boxes.append(boxes[fdx][bdx])
                    adjacent_scores += scores[fdx][bdx]
    
            # Calculate average box and score for missing frame
            if len(adjacent_boxes) == 0:
                print("ERROR")
                print(adjacent_boxes)
                print(missing_frame)
                print(sequence)
                exit()
            new_box = adjacent_boxes[0]
            new_score = adjacent_scores
            if len(adjacent_boxes) >= 2:
                new_box = calculate_avg_box(adjacent_boxes)
                new_score = adjacent_scores / len(adjacent_boxes)
            boxes_to_add.append((missing_frame, new_box, new_score))
        
        # print("Need to add these boxes to complete the frameset")
        # for frame_to_add, box_to_add in boxes_to_add:
        #     print("\tAdd box", box_to_add, "to frame #", frame_to_add)
        
        boxes_for_sequences.append(boxes_to_add)
    
    return boxes_for_sequences

def check_for_missing_links_in_sequences(sequences, boxes=[], scores=[]):
    missing_frames_sqx = []
    for sqx, sequence in enumerate(sequences):
        seq_max_frame = max(sequence, key=itemgetter(0))[0]
        seq_min_frame = min(sequence, key=itemgetter(0))[0]
        
        missing_frames = set([x for x in range(seq_min_frame, seq_max_frame + 1)])
        for frame, box in sequence:
            try:
                missing_frames.remove(frame)
            except:
                # print("FRAME that went bad", frame)
                # print(seq_min_frame, seq_max_frame)
                # print(missing_frames)
                # print("Sequence", sequence)
                # exit()
                # print("Potential duplicates in sequence", sqx, "with frame #", frame)
                pass
        missing_frames_sqx.append(missing_frames)
        print("Sequence #", sqx, "missing frames", missing_frames)

    return missing_frames_sqx

def find_sequences_graph_from_overlap(overlap_matrix, num_frames, num_boxes, scores,sequence_length_threshold=2, window_size=5, linkage_threshold=0.5):
    G = generate_weighted_directed_graph(overlap_matrix, num_frames, num_boxes)
    # IDEA Create the network graph from the overlap matrix instead of the best_edge matrix
    # Iteratively run through a maximum path algorithm
    # Start with node with highest score in f_0
    #   Find best sequence from f_0->f_1->f_2->f_3 until f_4 has no more connected points for the object
    #   Then delete the nodes from the network
    # Repeat until there are no more possible sequences

    scores_copy = np.copy(scores)
    print(scores)
    print(G)
    current_frame = 0
    sequences = []
    while(1):
        
        if current_frame == num_frames:
            break
        
        # If we have no active boxes in the current frame, increment the frame
        if np.max(scores_copy[current_frame]) == 0:
            # print("Scores remaining in last frame", scores_copy[current_frame])
            current_frame+=1
            # print("Current Frame looking at", current_frame)
            continue
    
        # Choose the box with the highest score to start building a sequence from the current frame
        start_box = np.argmax(scores_copy[current_frame])
        # print("\tStart box", start_box, "curr frame", current_frame)
        
        # Build the sequence
        result_seq = find_next_best_sequence(G, current_frame, start_box, scores_copy, num_frames, window_size=window_size, linkage_threshold=linkage_threshold)
        # print("\t\tFinished sequences", result_seq)
        for node in result_seq:
            G.remove_node(node)
            # print("\t\tNODE", node)
            scores_copy[node[0]][node[1]] = 0
        
        if len(result_seq) >= sequence_length_threshold:
            sequences.append(result_seq)
            
    return sequences

def find_next_best_sequence(G, start_frame, start_box, scores, num_frames, window_size=5, linkage_threshold=0.5):    
    current_node = G[(start_frame, start_box)]
    current_sequence = [(start_frame, start_box)]
    current_frame = start_frame
    looking_at_frame = current_frame + 1
    # print("\t\tStarting Node", current_node)
    while 1:
        frames_at_looking_frame = []
        # scores_at_looking_frame = []
        overlaps_at_looking_frame = []

        # If we get to the end of the frames
        if looking_at_frame > num_frames:
            # print("\t\t\tGot to end of all frames, breaking")
            break
        if looking_at_frame >= current_frame + window_size:
            # print("\t\t\tGot to end of window, breaking")
            break
        for edx, edge in enumerate(current_node):
            if edge[0] == looking_at_frame:
                # print("\t\t\tLooking at edge", edge, current_node[edge]['weight'])
                frames_at_looking_frame.append(edge)
                # scores_at_looking_frame.append(scores[edge[0]][edge[1]])
                overlaps_at_looking_frame.append(current_node[edge]['weight'])

        # If we found the next node in the sequence
        if len(frames_at_looking_frame) > 0:

            max_ovlp = overlaps_at_looking_frame[0]
            max_ovlp_idx = 0
            for tst_ovlp_idx, ovlp  in enumerate(overlaps_at_looking_frame):
                if ovlp > max_ovlp:
                    max_ovlp = ovlp
                    max_ovlp_idx = tst_ovlp_idx
                
                if ovlp == max_ovlp and len(frames_at_looking_frame[tst_ovlp_idx]) > len(frames_at_looking_frame[max_ovlp_idx]):
                    max_ovlp = ovlp
                    max_ovlp_idx = tst_ovlp_idx
            
            ovlp_idx = max_ovlp_idx

            # This is replaced by the code above to break ties by looking at the longest sequence
            # ovlp_idx = np.argmax(overlaps_at_looking_frame)


            choosen_edge = frames_at_looking_frame[ovlp_idx]
            current_frame+=1
            current_node = G[choosen_edge]
            current_sequence.append(choosen_edge)
            looking_at_frame+= 1
        # If we didn't find the next node in the sequence -> Skip a frame
        else:
            # print("\t\tNo next sequence found in frame",looking_at_frame, current_node)
            looking_at_frame += 1
    return current_sequence


def generate_weighted_directed_graph(overlap_matrix, num_frames, num_boxes)->networkx.Graph:
    G = networkx.Graph()
    for f in range(num_frames):
        for x in range(num_boxes):
            G.add_node((f,x))
    # print("DImensions of overlap matrix", overlap_matrix.shape)
    # Create a weighted directed graph
    for fdx, frame in enumerate(overlap_matrix):
        # frame = np.transpose(frame)
        # print("Frame #", fdx)
        for fjx, overlapped_frames in enumerate(frame):
            if fjx == fdx: 
                        continue
            # print("\tOverlap with Frame #", fjx)
            for bdx, overlapped_boxes in enumerate(overlapped_frames): 
                # print("\t\tBox #", bdx)
                for bjx, bjx_bdx_ovlp in enumerate(overlapped_boxes): # Overlap between Frame FDX, Box BDX and Frame FJX, Box BJX
                    # print("\t\t\twith box:", bjx, bjx_bdx_ovlp)
                    if bjx_bdx_ovlp > 0:
                        G.add_edge((fdx, bdx), (fjx, bjx), weight=bjx_bdx_ovlp)
    return G

# This works decently well but can end with some sequences being merged on accident. 
# This can be fixed by removing edges from a graph after they have been identified
# THis will fix it because
# A1 ->
#       B   can result in sequence A1, A2, B
# A2 ->
# When it should be A1 and seperate A2, B or vice versa
def find_sequences_graphs(best_edge_matrix, scores, num_frames, num_boxes, tiny_subgraphs=0, seq_confidence_threshold=0.5):
    # Create the nodes in the graph
    G = networkx.Graph()
    for f in range(num_frames):
        for x in range(num_boxes):
            G.add_node((f,x))
    
    # Add links in the graph
    for fdx, frame in enumerate(best_edge_matrix):
        # print("Frame #", fdx)
        frame = np.transpose(frame)
        for bdx, connected_boxes in enumerate(frame):
            # print("\tBox #", bdx)
            # print("\t\tConnected to ", connected_boxes)
            for fjx, bjx in enumerate(connected_boxes):
                if bjx == -1:
                    continue
                G.add_edge((fdx, bdx), (fjx, bjx))
    
    subgraphs = [G.subgraph(c).copy() for c in connected_components(G)]
    graph_nodes = [set([(int(y[0]),int(y[1])) for y in x.nodes]) for x in subgraphs if len(x.nodes) > tiny_subgraphs]

    output_nodes = []
    for node_list in graph_nodes:
        if np.max([scores[f_x][b_x] for (f_x, b_x) in node_list]) > seq_confidence_threshold:
            output_nodes.append(node_list)

    return output_nodes

def find_existing_sequences(best_edge_matrix, num_boxes, num_frames, boxes=[], scores=[], labels=[], should_print=False, sequence_threshold=0.7):
    existing_sequences = []

    for idx, best_edges in enumerate(best_edge_matrix):
        # Iterate across boxes instead of frames
        best_edges = np.transpose(best_edges)
        print("Frame ", idx)
        # print("\t", best_edges)
        for bdx, connected_boxes in enumerate(best_edges):
            # Create a set for all connected frames to this box
            box_set = set()
            for jdx, connected_box in enumerate(connected_boxes):
                if not connected_box == -1:
                    box_set.add((jdx, int(connected_box)))
                    best_edge_matrix[idx][jdx][bdx] = -1
                    best_edge_matrix[jdx][idx][bdx] = -1
            print("\tAll connected boxes to ", bdx)
            print("\t\t", box_set)
            existing_sequences.append(box_set)

    # Remove duplicate sets
    result = []
    for item in existing_sequences:
        if item not in result:
            result.append(item)

    for x in result:
        print(x)
    # Remove subsets
    l = result[:]
    l2 = result[:]
    for m in l:
        for n in l:
            if m.issubset(n) and m != n:
                l2.remove(m)
                break
    
    if should_print:
        for sqx, sequence in enumerate(result):
            print("Sequence #",sqx, "contains")
            for frame, box in sequence: 
                print("\tFrame", frame, "box#", box)
        
    return l2


# Generates a matrix where each node is connected to the best edges
def generate_best_edge_matrix(boxes, scores=[], labels=[], linkage_threshold=0.3, should_print=False, window_size=5):
    num_frames = boxes.shape[0]
    num_boxes = boxes.shape[1]

    overlap_matrix_np = np.zeros((num_frames, num_frames, num_boxes))
    for f_i in range(boxes.shape[0]):
        window_start = max(0, f_i - (window_size//2))
        window_end = min(num_frames - 1, f_i + (window_size//2))

        # Calculate the window if we are at the beginning
        if f_i - (window_size//2) < 0:
            window_end = min(num_frames, window_start + window_size - 1)
        
        # Calculate the window if we are at the end
        if f_i + (window_size//2) > num_frames - 1:
            window_start = max(0, window_end - window_size + 1)
        
        # boxes_f_i, scores_f_i = boxes[f_i,:,:], scores[f_i,:]
        boxes_f_i = boxes[f_i,:,:]
        frame_i_all_overlaps = np.full((num_frames, num_boxes), -1)
        if should_print:
            print("Frame ", f_i)
            print("\tWindow for frame [", window_start, ",",window_end,"]")
        # Iterate through all subsuquent frames within the window
        for f_j in range(window_start, window_end+1):
            if f_j == f_i:
                continue
            if should_print:
                print("\tComparing to frame", f_j)
            # if f_i == f_j:
            #     continue
            # boxes_f_j, scores_f_j = boxes[f_j,:,:], scores[f_j,:]
            boxes_f_j = boxes[f_j,:,:]
            # Calculate 2d overlap matrix between Frames I and J
            frame_ij_overlap = calculate_overlap_between_two_frames(boxes_f_i, boxes_f_j, linkage_threshold)
            best_overlaps_i_j = np.zeros(num_boxes)
            if(should_print):
                print("\t Overlaps between frames")
                print("", frame_ij_overlap)
            # iterate through each box and find the best matching overlap in the subsequent frames
            for bdx, box in enumerate(frame_ij_overlap):
                best_overlap = -1
                if np.max(box) >= linkage_threshold:
                    best_overlap = np.argmax(box)
                print("\t\tbdx", bdx, "best overlaps with", best_overlap)
                best_overlaps_i_j[bdx] = best_overlap
            frame_i_all_overlaps[f_j] = best_overlaps_i_j
        overlap_matrix_np[f_i] = frame_i_all_overlaps
    return overlap_matrix_np

# Generates a general overlap matrix
def generate_overlap_matrix(boxes, scores=[], labels=[], linkage_threshold=0.3, window_size=5):
    num_frames = boxes.shape[0]
    num_boxes = boxes.shape[1]

    overlap_matrix_np = np.zeros((num_frames, num_frames, num_boxes, num_boxes), np.float)
    # Iterate through all frames
    for f_i in range(boxes.shape[0]):
        boxes_f_i = boxes[f_i,:,:]
        frame_i_all_overlaps = np.zeros((num_frames, num_boxes, num_boxes), np.float)

        window_start = max(0, f_i - (window_size//2))
        window_end = min(num_frames - 1, f_i + (window_size//2))

        # Calculate the window if we are at the beginning
        if f_i - (window_size//2) < 0:
            window_end = min(num_frames, window_start + window_size - 1)
        
        # Calculate the window if we are at the end
        if f_i + (window_size//2) > num_frames - 1:
            window_start = max(0, window_end - window_size + 1)

        # For each Frame I iterate through all other frames within the window
        for f_j in range(window_start, window_end+1):
            # Skip when the frames are the same
            if f_i == f_j:
                continue

            # Isolate the boxes for frame j
            boxes_f_j = boxes[f_j,:,:]

            # Calculate overlap between frame I and J
            frame_ij_overlap = calculate_overlap_between_two_frames(boxes_f_i, boxes_f_j, linkage_threshold)
            frame_i_all_overlaps[f_j] = frame_ij_overlap 

        overlap_matrix_np[f_i] = frame_i_all_overlaps

    return overlap_matrix_np

def calculate_overlap_between_two_frames(boxes_f, boxes_f1, linkage_threshold):
    areas_f1 = compute_area(boxes_f1.astype(np.double)) 
    adjacency_matrix = np.zeros((len(boxes_f), len(boxes_f)))
    for i, box in enumerate(boxes_f):
        overlaps = compute_overlap_areas_given(np.expand_dims(box,axis=0).astype(np.double), boxes_f1.astype(np.double), areas_f1.astype(np.double) )[0]
        adjacency_matrix[i] = [x if x>linkage_threshold else 0 for x in overlaps]

    return adjacency_matrix

# TODO
def calculate_avg_box(boxes):
    return boxes[0]

if __name__ == '__main__':

    # Load the boxes so we don't have wait for neural network stuff when doing testing    
    boxes = np.load("test_cows_box.npy")
    scores = np.load("test_cows_score.npy")
    labels = np.load("test_cows_label.npy")

    print(boxes)
    print(scores)
    print(labels)

    build_box_sequences(boxes, scores, labels)
