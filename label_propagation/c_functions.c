#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <unistd.h>


#define TOP_COUNT 50
#define TOP_POWER 0.4
#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))


// get_active_map() function in C
unsigned char* get_active_map(int subtitles, int faces, int roles,
                              long long* subtitle_timestamp, long long* face_timestamp, int* subtitle_code, int* face_code,
                              float* face_score, unsigned char* mask, int face_radius, float face_threshold) {
    unsigned char* active_map = (unsigned char*)malloc(sizeof(unsigned char) * subtitles * roles);
    memset(active_map, 0, sizeof(unsigned char) * subtitles * roles);
    int* role_count = (int*)malloc(sizeof(int) * roles);
    memset(role_count, 0, sizeof(int) * roles);
    int left_face_idx = 0;
    int right_face_idx = 0;
    int left_subtitle_idx = 0;
    int right_subtitle_idx = 0;
    for (int subtitle_idx = 0; subtitle_idx < subtitles; subtitle_idx++) {
        // check all face occurrences
        while (right_face_idx < faces &&
            face_timestamp[right_face_idx * 2] <= subtitle_timestamp[subtitle_idx * 2 + 1] + face_radius) {
            if (0 <= face_code[right_face_idx] && face_code[right_face_idx] < roles &&
                face_score[right_face_idx] >= face_threshold) {
                role_count[face_code[right_face_idx]]++;
            }
            right_face_idx++;
        }
        while (left_face_idx < faces &&
            face_timestamp[left_face_idx * 2 + 1] < subtitle_timestamp[subtitle_idx * 2] - face_radius) {
            if (0 <= face_code[left_face_idx] && face_code[left_face_idx] < roles &&
                face_score[left_face_idx] >= face_threshold) {
                role_count[face_code[left_face_idx]]--;
            }
            left_face_idx++;
        }
        // check all subtitle (with speaker) occurrences
        while (right_subtitle_idx < subtitles &&
            subtitle_timestamp[right_subtitle_idx * 2] <= subtitle_timestamp[subtitle_idx * 2 + 1] + face_radius) {
            if (0 <= subtitle_code[right_subtitle_idx] && subtitle_code[right_subtitle_idx] < roles) {
                role_count[subtitle_code[right_subtitle_idx]]++;
            }
            right_subtitle_idx++;
        }
        while (left_subtitle_idx < subtitles &&
            subtitle_timestamp[left_subtitle_idx * 2 + 1] < subtitle_timestamp[subtitle_idx * 2] - face_radius) {
            if (0 <= subtitle_code[left_subtitle_idx] && subtitle_code[left_subtitle_idx] < roles) {
                role_count[subtitle_code[left_subtitle_idx]]--;
            }
            left_subtitle_idx++;
        }
        if (mask[subtitle_idx] == 1) {
            long long array_address = (long long)subtitle_idx * roles;
            for (int role_idx = 0; role_idx < roles; role_idx++) {
                active_map[array_address] = role_count[role_idx] > 0 ? 1 : 0;
                array_address++;
            }
        }
    }
    free(role_count);
    return active_map;
}

// get_connected_components() function in C
int* get_connected_components(int subtitles, int active_subtitles,
                              float* similarity, int* active_idx,
                              float sim_threshold, int search_radius) {
    float (*similarity_2d)[subtitles] = (float (*)[subtitles])similarity;
    int* start_pos = NULL;
    int* end_pos = NULL;
    // initialize the search positions
    if (search_radius > 0) {
        start_pos = (int*)malloc(sizeof(int) * active_subtitles);
        end_pos = (int*)malloc(sizeof(int) * active_subtitles);
        memset(start_pos, 0, sizeof(int) * active_subtitles);
        memset(end_pos, 0, sizeof(int) * active_subtitles);
        int left_cursor = 0;
        int right_cursor = 0;
        for (int subtitle_idx = 0; subtitle_idx < active_subtitles; subtitle_idx++) {
            while (active_idx[left_cursor] < active_idx[subtitle_idx] - search_radius) {
                left_cursor++;
            }
            while (right_cursor < active_subtitles && active_idx[right_cursor] <= active_idx[subtitle_idx] + search_radius) {
                right_cursor++;
            }
            start_pos[subtitle_idx] = left_cursor;
            end_pos[subtitle_idx] = right_cursor;
        }
    }
    // perform floodfill to find connected components
    int* result = (int*)malloc(sizeof(int) * active_subtitles * 2);
    memset(result, -1, sizeof(int) * active_subtitles * 2);
    int components = 0;
    int* queue =  (int*)malloc(sizeof(int) * active_subtitles);
    for (int subtitle_idx = 0; subtitle_idx < active_subtitles; subtitle_idx++) {
        if (result[subtitle_idx] == -1) {
            int head = 0;
            int tail = 1;
            queue[0] = subtitle_idx;
            result[subtitle_idx] = components;
            while (head < tail) {
                int start_idx = search_radius > 0 ? max(subtitle_idx, start_pos[queue[head]]) : subtitle_idx;
                int end_idx = search_radius > 0 ? min(active_subtitles, end_pos[queue[head]]) : active_subtitles;
                for (int next_idx = start_idx; next_idx < end_idx; next_idx++) {
                    //long long array_address = (long long)active_idx[queue[head]] * subtitles + active_idx[next_idx];
                    if (result[next_idx] == -1 && similarity_2d[active_idx[queue[head]]][active_idx[next_idx]] >= sim_threshold) {
                        queue[tail] = next_idx;
                        tail++;
                        result[next_idx] = components;
                    }
                }
                head++;
            }
            result[active_subtitles + components] = tail;
            components++;
        }
    }
    if (search_radius > 0) {
        free(start_pos);
        free(end_pos);
    }
    free(queue);
    return result;
}

// quicksort: replaced by quickselect and not used in the current version
void quicksort(float* buffer, int low_idx, int high_idx) {
    if (low_idx < high_idx) {
        float pivot = buffer[high_idx];
        int middle_idx = low_idx;
        for (int right_idx = low_idx; right_idx < high_idx; right_idx++) {
            if (buffer[right_idx] >= pivot) {
                float temp = buffer[middle_idx];
                buffer[middle_idx] = buffer[right_idx];
                buffer[right_idx] = temp;
                middle_idx++;
            }
        }
        float temp = buffer[middle_idx];
        buffer[middle_idx] = buffer[high_idx];
        buffer[high_idx] = temp;
        quicksort(buffer, low_idx, middle_idx - 1);
        quicksort(buffer, middle_idx + 1, high_idx);
    }
}

// non-strict quickselect: can guarantee top_indices elements to be maximum ones
void quickselect(float* buffer, int low_idx, int high_idx, int top_indices) {
    float pivot = buffer[high_idx];
    int middle_idx = low_idx;
    for (int right_idx = low_idx; right_idx < high_idx; right_idx++) {
        if (buffer[right_idx] >= pivot) {
            float temp = buffer[middle_idx];
            buffer[middle_idx] = buffer[right_idx];
            buffer[right_idx] = temp;
            middle_idx++;
        }
    }
    float temp = buffer[middle_idx];
    buffer[middle_idx] = buffer[high_idx];
    buffer[high_idx] = temp;
    int low_count = middle_idx - low_idx;
    if (top_indices < low_count) {
        if (low_idx < middle_idx - 1) {
            quickselect(buffer, low_idx, middle_idx - 1, top_indices);
        }
    } else if (top_indices > low_count + 1) {
        if (middle_idx < high_idx) {
            quickselect(buffer, middle_idx + 1, high_idx, top_indices - (low_count + 1));
        }
    }
}

// strict quickselect: beyond quickselect(), can guarantee the pivot to be at the position #top_indices
// while it can be merged into quickselect() with a parameter, we keep this version for efficiency
void quickselect_strict(float* buffer, int low_idx, int high_idx, int top_indices) {
    float pivot = buffer[high_idx];
    int middle_idx = low_idx;
    for (int right_idx = low_idx; right_idx < high_idx; right_idx++) {
        if (buffer[right_idx] >= pivot) {
            float temp = buffer[middle_idx];
            buffer[middle_idx] = buffer[right_idx];
            buffer[right_idx] = temp;
            middle_idx++;
        }
    }
    float temp = buffer[middle_idx];
    buffer[middle_idx] = buffer[high_idx];
    buffer[high_idx] = temp;
    int low_count = middle_idx - low_idx;
    if (top_indices < low_count) {
        if (low_idx < middle_idx - 1) {
            quickselect(buffer, low_idx, middle_idx - 1, top_indices);
        }
    } else if (top_indices > low_count) {
        if (middle_idx < high_idx) {
            quickselect(buffer, middle_idx + 1, high_idx, top_indices - (low_count + 1));
        }
    }
}

// get_propagated_labels() function in C
float* get_propagated_labels(int subtitles, int roles, int active_subtitles,
                             float* similarity, unsigned char* active_map, int* active_idx, int* current_label,
                             int* subtitle_length, float sim_threshold, int max_candidates) {
    float (*similarity_2d)[subtitles] = (float (*)[subtitles])similarity;
    unsigned char (*active_map_2d)[roles] = (unsigned char (*)[roles])active_map;
    // compute role active idx linklist for fast enumeration
    int* role_active_idx_first = (int*)malloc(sizeof(int) * roles);
    memset(role_active_idx_first, -1, sizeof(int) * roles);
    int* role_active_idx_last = (int*)malloc(sizeof(int) * roles);
    memset(role_active_idx_last, -1, sizeof(int) * roles);
    int** role_active_idx_prev = (int**)malloc(sizeof(int*) * roles);
    int** role_active_idx_next = (int**)malloc(sizeof(int*) * roles);
    for (int role_idx = 0; role_idx < roles; role_idx++) {
        role_active_idx_prev[role_idx] = (int*)malloc(sizeof(int) * subtitles);
        memset(role_active_idx_prev[role_idx], -1, sizeof(int) * subtitles);
        role_active_idx_next[role_idx] = (int*)malloc(sizeof(int) * subtitles);
        memset(role_active_idx_next[role_idx], -1, sizeof(int) * subtitles);
        for (int active_subtitle_idx = 0; active_subtitle_idx < active_subtitles; active_subtitle_idx++) {
            int subtitle_idx = active_idx[active_subtitle_idx];
//            long long array_address = (long long)subtitle_idx * roles + role_idx;
//            if (active_map[array_address] == 1) {
            if (active_map_2d[subtitle_idx][role_idx] == 1) {
                if (role_active_idx_first[role_idx] == -1) {
                    role_active_idx_first[role_idx] = subtitle_idx;
                    role_active_idx_last[role_idx] = subtitle_idx;
                } else {
                    int last_idx = role_active_idx_last[role_idx];
                    role_active_idx_prev[role_idx][subtitle_idx] = last_idx;
                    role_active_idx_next[role_idx][last_idx] = subtitle_idx;
                    role_active_idx_last[role_idx] = subtitle_idx;
                }
            }
        }
    }
    // create role active list for update
    int* role_labeled_indices = (int*)malloc(sizeof(int) * roles);
    memset(role_labeled_indices, 0, sizeof(int) * roles);
    int** role_labeled_idx = (int**)malloc(sizeof(int*) * roles);
    for (int role_idx = 0; role_idx < roles; role_idx++) {
        role_labeled_idx[role_idx] = (int*)malloc(sizeof(int) * subtitles);
        memset(role_labeled_idx[role_idx], -1, sizeof(int) * subtitles);
    }
    int* max_subtitle_length = (int*)malloc(sizeof(int) * roles);
    memset(max_subtitle_length, 0, sizeof(int) * roles);
    for (int subtitle_idx = 0; subtitle_idx < subtitles; subtitle_idx++) {
        int role_idx = current_label[subtitle_idx];
        if (role_idx >= 0) {
            role_labeled_idx[role_idx][role_labeled_indices[role_idx]] = subtitle_idx;
            role_labeled_indices[role_idx]++;
            max_subtitle_length[role_idx] = max(max_subtitle_length[role_idx], subtitle_length[subtitle_idx]);
        }
    }
    // get valid labeled subtitles (by length)
    int max_length_threshold = 800;
    int unit_length_threshold = 200;
    int* role_valid_indices = (int*)malloc(sizeof(int) * roles);
    memset(role_valid_indices, 0, sizeof(int) * roles);
    int** role_valid_idx = (int**)malloc(sizeof(int*) * roles);
    for (int role_idx = 0; role_idx < roles; role_idx++) {
        role_valid_idx[role_idx] = (int*)malloc(sizeof(int) * subtitles);
        int length_threshold = min(max_subtitle_length[role_idx] / unit_length_threshold * unit_length_threshold, max_length_threshold);
        for (int role_subtitle_idx = 0; role_subtitle_idx < role_labeled_indices[role_idx]; role_subtitle_idx++) {
            int subtitle_idx = role_labeled_idx[role_idx][role_subtitle_idx];
            if (subtitle_length[subtitle_idx] >= length_threshold) {
                role_valid_idx[role_idx][role_valid_indices[role_idx]] = subtitle_idx;
                role_valid_indices[role_idx]++;
            }
        }
    }
    float** sim_avg = (float**)malloc(sizeof(float*) * subtitles);
    float* subtitle_added_similarity = (float*)malloc(sizeof(float) * subtitles);
    float* subtitle_sim_max = (float*)malloc(sizeof(float) * subtitles);
    for (int subtitle_idx = 0; subtitle_idx < subtitles; subtitle_idx++) {
        sim_avg[subtitle_idx] = (float*)malloc(sizeof(float) * roles);
        memset(sim_avg[subtitle_idx], 0, sizeof(float) * roles);
        subtitle_added_similarity[subtitle_idx] = min(max(1 - (float)subtitle_length[subtitle_idx] / 1000, 0), 0.5) * 0.2;
        subtitle_sim_max[subtitle_idx] = -1;
    }
    unsigned char* role_unchanged = (unsigned char*)malloc(sizeof(unsigned char) * roles);
    memset(role_unchanged, 0, sizeof(unsigned char) * roles);
    // perform label propagation
    int new_labels = 0;
    int unit_size = 1 + max_candidates * 2;
    float* new_label_list = (float*)malloc(sizeof(float) * active_subtitles * unit_size);
    memset(new_label_list, 0, sizeof(float) * active_subtitles * unit_size);
    float* sim_buffer = (float*)malloc(sizeof(float) * subtitles);
    memset(sim_buffer, 0, sizeof(float) * subtitles);
    while (true) {
        for (int role_idx = 0; role_idx < roles; role_idx++) {
            if (role_unchanged[role_idx] || role_labeled_indices[role_idx] == 0 || role_active_idx_first[role_idx] == -1) {
                continue;
            }
            // compute average similarity
            int subtitle_idx = role_active_idx_first[role_idx];
            while (subtitle_idx >= 0) {
                for (int role_subtitle_idx = 0; role_subtitle_idx < role_valid_indices[role_idx]; role_subtitle_idx++) {
//                    long long array_address = (long long)role_valid_idx[role_idx][role_subtitle_idx] * subtitles + subtitle_idx;
//                    sim_buffer[role_subtitle_idx] = similarity[array_address];
                    sim_buffer[role_subtitle_idx] = similarity_2d[role_valid_idx[role_idx][role_subtitle_idx]][subtitle_idx];
                }
//                int top_indices = min(role_valid_indices[role_idx], max(16, min(64, floor(pow(role_valid_indices[role_idx], 0.4)))));
                int top_indices = min(TOP_COUNT, floor(pow(role_valid_indices[role_idx], TOP_POWER)));
                if (role_valid_indices[role_idx] > top_indices) {
                    quickselect(sim_buffer, 0, role_valid_indices[role_idx] - 1, top_indices);
                }
                float sim_sum = 0;
                for (int role_subtitle_idx = 0; role_subtitle_idx < top_indices; role_subtitle_idx++) {
                    sim_sum += sim_buffer[role_subtitle_idx];
                }
                sim_avg[subtitle_idx][role_idx] = sim_sum / top_indices + subtitle_added_similarity[subtitle_idx];
                subtitle_sim_max[subtitle_idx] = max(subtitle_sim_max[subtitle_idx], sim_avg[subtitle_idx][role_idx]);
                subtitle_idx = role_active_idx_next[role_idx][subtitle_idx];
            }
            role_unchanged[role_idx] = true;
        }
        int new_labels_iter = 0;
        for (int active_subtitle_idx = 0; active_subtitle_idx < active_subtitles; active_subtitle_idx++) {
            int subtitle_idx = active_idx[active_subtitle_idx];
            if (subtitle_sim_max[subtitle_idx] < sim_threshold) {
                continue;
            }
            int active_roles = 0;
            int offset = new_labels * unit_size;
            new_label_list[offset] = subtitle_idx;
            // TODO: maybe using a linklist for acceleration?
            for (int role_idx = 0; role_idx < roles; role_idx++) {
                if (sim_avg[subtitle_idx][role_idx] >= sim_threshold) {
                    int insert_idx = active_roles;
                    while (insert_idx > 0 && sim_avg[subtitle_idx][role_idx] > new_label_list[2 * insert_idx + offset]) {
                        insert_idx--;
                    }
                    if (insert_idx == max_candidates) {
                        continue;
                    }
                    for (int op_idx = min(active_roles, max_candidates) - 1; op_idx > insert_idx; op_idx--) {
                        new_label_list[2 * op_idx + offset + 1] = new_label_list[2 * op_idx + offset - 1];
                        new_label_list[2 * op_idx + offset + 2] = new_label_list[2 * op_idx + offset];
                    }
                    new_label_list[2 * insert_idx + offset + 1] = role_idx;
                    new_label_list[2 * insert_idx + offset + 2] = sim_avg[subtitle_idx][role_idx];
                    if (active_roles < max_candidates) {
                        active_roles++;
                    }
                }
            }
            for (int role_idx = 0; role_idx < roles; role_idx++) {
//                long long array_address = (long long)subtitle_idx * roles + role_idx;
//                if (active_map[array_address] == 1) {
                if (active_map_2d[subtitle_idx][role_idx] == 1) {
//                    active_map[array_address] = 0;
                    active_map_2d[subtitle_idx][role_idx] = 0;
                    if (role_active_idx_first[role_idx] == subtitle_idx) {
                        if (role_active_idx_last[role_idx] == subtitle_idx) {
                            role_active_idx_first[role_idx] = -1;
                            role_active_idx_last[role_idx] = -1;
                        } else {
                            int next_subtitle_idx = role_active_idx_next[role_idx][subtitle_idx];
                            role_active_idx_first[role_idx] = next_subtitle_idx;
                            role_active_idx_prev[role_idx][next_subtitle_idx] = -1;
                            role_active_idx_next[role_idx][subtitle_idx] = -1;
                        }
                    } else {
                        if (role_active_idx_last[role_idx] == subtitle_idx) {
                            int prev_subtitle_idx = role_active_idx_prev[role_idx][subtitle_idx];
                            role_active_idx_last[role_idx] = prev_subtitle_idx;
                            role_active_idx_next[role_idx][prev_subtitle_idx] = -1;
                            role_active_idx_prev[role_idx][subtitle_idx] = -1;
                        } else {
                            int prev_subtitle_idx = role_active_idx_prev[role_idx][subtitle_idx];
                            int next_subtitle_idx = role_active_idx_next[role_idx][subtitle_idx];
                            role_active_idx_next[role_idx][prev_subtitle_idx] = next_subtitle_idx;
                            role_active_idx_prev[role_idx][next_subtitle_idx] = prev_subtitle_idx;
                            role_active_idx_prev[role_idx][subtitle_idx] = -1;
                            role_active_idx_next[role_idx][subtitle_idx] = -1;
                        }
                    }
                }
            }
            int role_idx = (int)new_label_list[offset + 1];
            role_unchanged[role_idx] = false;
            role_labeled_idx[role_idx][role_labeled_indices[role_idx]] = subtitle_idx;
            role_labeled_indices[role_idx]++;
            if (max_subtitle_length[role_idx] < subtitle_length[subtitle_idx]) {
                if (role_valid_indices[role_idx] > 0 && max_subtitle_length[role_idx] < max_length_threshold) {
                    int old_unit = max_subtitle_length[role_idx] / unit_length_threshold;
                    int new_unit = subtitle_length[subtitle_idx] / unit_length_threshold;
                    if (old_unit < new_unit) {
                        role_valid_indices[role_idx] = 0;
                    }
                }
                max_subtitle_length[role_idx] = subtitle_length[subtitle_idx];
            }
            int length_threshold = min(max_subtitle_length[role_idx] / unit_length_threshold * unit_length_threshold, max_length_threshold);
            if (subtitle_length[subtitle_idx] >= length_threshold) {
                role_valid_idx[role_idx][role_valid_indices[role_idx]] = subtitle_idx;
                role_valid_indices[role_idx]++;
            }
            subtitle_sim_max[subtitle_idx] = -1;
            new_labels++;
            new_labels_iter++;
        }
        if (new_labels_iter < 5) {
            break;
        }
    }
    // free all arrays
    free(role_active_idx_first);
    free(role_active_idx_last);
    for (int role_idx = 0; role_idx < roles; role_idx++) {
        free(role_active_idx_prev[role_idx]);
        free(role_active_idx_next[role_idx]);
    }
    free(role_active_idx_prev);
    free(role_active_idx_next);
    free(role_labeled_indices);
    for (int role_idx = 0; role_idx < roles; role_idx++) {
        free(role_labeled_idx[role_idx]);
    }
    free(role_labeled_idx);
    free(max_subtitle_length);
    free(role_valid_indices);
    for (int role_idx = 0; role_idx < roles; role_idx++) {
        free(role_valid_idx[role_idx]);
    }
    free(role_valid_idx);
    for (int subtitle_idx = 0; subtitle_idx < subtitles; subtitle_idx++) {
        free(sim_avg[subtitle_idx]);
    }
    free(sim_avg);
    free(subtitle_added_similarity);
    free(subtitle_sim_max);
    free(role_unchanged);
    free(sim_buffer);
    // return with protection
    if (new_labels < active_subtitles) {
        new_label_list[new_labels * unit_size] = -1;
    }
    return new_label_list;
}

// get the maximum element from an array
void get_max(float* buffer, int length, float* result_max) {
    float all_max = buffer[0];
    for (int buffer_idx = 1; buffer_idx < length; buffer_idx++) {
        if (buffer[buffer_idx] > all_max) {
            all_max = buffer[buffer_idx];
        }
    }
    *result_max = all_max;
}

// get the maximum and median elements simultaneously from an array (optimized)
void get_max_and_median(float* buffer, int length, float* result_max, float* result_median) {
    if (length > 1) {
        quickselect_strict(buffer, 0, length - 1, length / 2);
    }
    float top_half_max = buffer[0];
    if (length % 2 == 1) {
        for (int buffer_idx = 1; buffer_idx < length / 2; buffer_idx++) {
            if (buffer[buffer_idx] > top_half_max) {
                top_half_max = buffer[buffer_idx];
            }
        }
        *result_median = buffer[length / 2];
    } else {
        float top_half_min = buffer[0];
        for (int buffer_idx = 1; buffer_idx < length / 2; buffer_idx++) {
            if (buffer[buffer_idx] > top_half_max) {
                top_half_max = buffer[buffer_idx];
            } else if (buffer[buffer_idx] < top_half_min) {
                top_half_min = buffer[buffer_idx];
            }
        }
        *result_median = (buffer[length / 2] + top_half_min) / 2;
    }
    *result_max = top_half_max;
}

unsigned char is_high_similarity_(float active_idx_ratio, float mean_top, float base_threshold, float delta_threshold) {
    return (active_idx_ratio >= 0.5 && mean_top >= base_threshold) || (active_idx_ratio > 0 && mean_top >= base_threshold + delta_threshold);
}

// compute_rel_info() function in C
float* compute_rel_info(int subtitles, int roles, int components, int total_length,
                        float* similarity, unsigned char* active_map, int* current_label, int* idx_list_length, int* idx_list_all,
                        float base_threshold, float delta_threshold, unsigned char display_info) {
    float (*similarity_2d)[subtitles] = (float (*)[subtitles])similarity;
    unsigned char (*active_map_2d)[roles] = (unsigned char (*)[roles])active_map;
    int* role_labeled_idx_first = (int*)malloc(sizeof(int) * roles);
    memset(role_labeled_idx_first, -1, sizeof(int) * roles);
    int* role_labeled_idx_last = (int*)malloc(sizeof(int) * roles);
    memset(role_labeled_idx_last, -1, sizeof(int) * roles);
    int* role_labeled_idx_next = (int*)malloc(sizeof(int) * subtitles);
    memset(role_labeled_idx_next, -1, sizeof(int) * subtitles);
    for (int subtitle_idx = 0; subtitle_idx < subtitles; subtitle_idx++) {
        if (current_label[subtitle_idx] >= 0) {
            int role_idx = current_label[subtitle_idx];
            if (role_labeled_idx_first[role_idx] == -1) {
                role_labeled_idx_first[role_idx] = subtitle_idx;
            } else {
                role_labeled_idx_next[role_labeled_idx_last[role_idx]] = subtitle_idx;
            }
            role_labeled_idx_last[role_idx] = subtitle_idx;
        }
    }
    float* col_max = (float*)malloc(sizeof(float) * total_length);
    float* col_max_copy = (float*)malloc(sizeof(float) * total_length);
    float* col_median = (float*)malloc(sizeof(float) * total_length);
    float* avg_top = (float*)malloc(sizeof(float) * total_length);
    float* sim_buffer = (float*)malloc(sizeof(float) * subtitles);
    int unit_size = components * roles;
    float* result = (float*)malloc(sizeof(float) * (unit_size * 6 + total_length * 2));
    memset(result, 0, sizeof(float) * (unit_size * 6 + total_length * 2));
    float* active_idx_count = &result[0];
    float* max_max = &result[unit_size];
    float* max_median = &result[unit_size * 2];
    float* median_max = &result[unit_size * 3];
    float* mean_avg_top = &result[unit_size * 4];
    float* is_high_similarity = &result[unit_size * 5];
    float* global_sim = &result[unit_size * 6];
    for (int role_idx = 0; role_idx < roles; role_idx++) {
        if (role_labeled_idx_first[role_idx] == -1) {
            continue;
        }
        for (int idx_list_idx = 0; idx_list_idx < total_length; idx_list_idx++) {
            int subtitle_idx = idx_list_all[idx_list_idx];
            int role_subtitle_idx = role_labeled_idx_first[role_idx];
            int role_subtitle_count = 0;
            while (role_subtitle_idx >= 0) {
                sim_buffer[role_subtitle_count] = similarity_2d[subtitle_idx][role_subtitle_idx];
                role_subtitle_idx = role_labeled_idx_next[role_subtitle_idx];
                role_subtitle_count++;
            }
            get_max_and_median(sim_buffer, role_subtitle_count, &col_max[idx_list_idx], &col_median[idx_list_idx]);
            int top_indices = min(TOP_COUNT, floor(pow(role_subtitle_count, TOP_POWER)));
            if (role_subtitle_count > top_indices) {
                quickselect(sim_buffer, 0, role_subtitle_count - 1, top_indices);
            }
            float sim_sum = 0;
            for (int role_subtitle_idx = 0; role_subtitle_idx < top_indices; role_subtitle_idx++) {
                sim_sum += sim_buffer[role_subtitle_idx];
            }
            avg_top[idx_list_idx] = sim_sum / top_indices;
        }
        memcpy(col_max_copy, col_max, sizeof(float) * total_length);
        int offset = 0;
        for (int cp_idx = 0; cp_idx < components; cp_idx++) {
            int unit_idx = cp_idx * roles + role_idx;
            for (int idx_list_idx = offset; idx_list_idx < offset + idx_list_length[cp_idx]; idx_list_idx++) {
                int subtitle_idx = idx_list_all[idx_list_idx];
                if (active_map_2d[subtitle_idx][role_idx] == 1) {
                    active_idx_count[unit_idx]++;
                }
                mean_avg_top[unit_idx] += avg_top[idx_list_idx];
            }
            mean_avg_top[unit_idx] /= idx_list_length[cp_idx];
            get_max_and_median(&col_max[offset], idx_list_length[cp_idx], &max_max[unit_idx], &median_max[unit_idx]);
            get_max(&col_median[offset], idx_list_length[cp_idx], &max_median[unit_idx]);
            is_high_similarity[unit_idx] = is_high_similarity_((float)active_idx_count[unit_idx] / idx_list_length[cp_idx],
                mean_avg_top[unit_idx], base_threshold, delta_threshold);
            if (display_info) {
                int unit_idx_sim = offset * 2;
                for (int idx_list_idx = offset; idx_list_idx < offset + idx_list_length[cp_idx]; idx_list_idx++) {
                    if (is_high_similarity[unit_idx]) {
                        global_sim[unit_idx_sim] = max(global_sim[unit_idx_sim], col_max_copy[idx_list_idx]);
                    }
                    unit_idx_sim++;
                    global_sim[unit_idx_sim] = max(global_sim[unit_idx_sim], col_max_copy[idx_list_idx]);
                    unit_idx_sim++;
                }
            }
            offset += idx_list_length[cp_idx];
        }
    }
    free(role_labeled_idx_first);
    free(role_labeled_idx_last);
    free(role_labeled_idx_next);
    free(col_max);
    free(col_max_copy);
    free(col_median);
    free(avg_top);
    free(sim_buffer);
    return result;
}
