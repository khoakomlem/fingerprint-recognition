class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, label, x_real, label_real_dict, batch_size=32, shuffle=True):
        "Initialization"
        self.x = x
        self.label = label
        self.x_real = x_real
        self.label_real_dict = label_real_dict

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        x1_batch = self.x[index * self.batch_size : (index + 1) * self.batch_size]
        label_batch = self.label[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        x2_batch = np.empty((self.batch_size, 90, 90, 1), dtype=np.float32)
        y_batch = np.zeros((self.batch_size, 1), dtype=np.float32)

        # augmentation
        if self.shuffle:
            seq = iaa.Sequential(
                [
                    iaa.GaussianBlur(sigma=(0, 0.5)),
                    iaa.Affine(
                        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        rotate=(-30, 30),
                        order=[0, 1],
                        cval=255,
                    ),
                ],
                random_order=True,
            )

            x1_batch = seq.augment_images(x1_batch)
            # for img in x1_batch:
            #     img = Gabor(img)

        # pick matched images(label 1.0) and unmatched images(label 0.0) and put together in batch
        # matched images must be all same, [subject_id(3), gender(1), left_right(1), finger(1)], e.g) 034010
        for i, l in enumerate(label_batch):
            match_key = l.astype(str)
            match_key = "".join(match_key).zfill(6)

            if random.random() > 0.5:
                # put matched image
                x2_batch[i] = self.x_real[self.label_real_dict[match_key]]
                y_batch[i] = 1.0
            else:
                # put unmatched image
                while True:
                    unmatch_key, unmatch_idx = random.choice(
                        list(self.label_real_dict.items())
                    )

                    if unmatch_key != match_key:
                        break

                x2_batch[i] = self.x_real[unmatch_idx]
                y_batch[i] = 0.0

        return [
            x1_batch.astype(np.float32) / 255.0,
            x2_batch.astype(np.float32) / 255.0,
        ], y_batch

    def on_epoch_end(self):
        if self.shuffle == True:
            self.x, self.label = shuffle(self.x, self.label)
