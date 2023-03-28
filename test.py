arr = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 1])
differences = arr[1:]- arr[:-1]
start_idxes = np.where(differences > 0)[0]+1
if arr[0] == 1:
    start_idxes = np.insert(start_idxes, 0, 0, axis=0)
end_idxes = np.where(differences < 0)[0]
if arr[-1] == 1:
    end_idxes = np.append(end_idxes,len(arr)-1)
