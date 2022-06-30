# dfuc22
W pliku training_template.py i eval_template.py znajdują się przykłady generowania danych dla uczenia i ewaluacji. Chodzi o to, żebyśmy dla każdego foldu, mieli te same pliki, a testset był zawsze taki sam.

W pliku eval_template.py znajduje się też funkcja licząca dice, obecnie jest to dice per image (dla uproszczenia), można się zastanowić nad implementacją per zmiana. Ale przede wszystkim ważne, żebyśmy mieli tak samo liczoną metrykę.

Dodatkowo, ewaluacja przewidziana jest na N modeli (gdzie N to liczba foldów - przyjmujemy 5). Strategia ewaluacji zależy od Was (ja uśredniam głosy każdego z modeli na test secie). Najlepiej by było jakbyście uczyli również wszystkie 5 modeli i je wykorzystali w ewaluacji. Jeśli jednak z jakiś powodów (czas uczenia - choć na plgridzie można równolegle te foldy uczyć) będzie to jeden model (1 fold) zaznaczcie to w opisie modelu w tabeli poniżej.

Aby uruchomić ewaluację, należy wywołać skrypt z argumentami:<br />
eval_template.py folds_count imgs_dir masks_dir

na przykład:<br />
5
/absolute_path_to/imgs/DFUC2022_train_images
/absolute_path_to/masks/DFUC2022_train_masks

w przypadku uczenia: <br />
training_template.py fold_no folds_count imgs_dir masks_dir <br />

na przykład:<br />
3
5
/home/darekk/dev/dfuc2022_challenge/DFUC2022_train_release/DFUC2022_train_images
/home/darekk/dev/dfuc2022_challenge/DFUC2022_train_release/DFUC2022_train_masks

folds_count - liczba foldów, przyjmujemy 5<br />
fold_no - w zależności od tego, który fold uczycie (0-4; dla eval, nie ma znaczenia)<br />


Pliki fold_x_out_of_5_debug_files_list.json zawierają takie dla każdego folda i pełnią rolę informacyjną. Generują się po ustawieniu argumentu save_debug_file na True funkcji get_foldwise_split (przykładowo w training_template.py). Gdyby z jakiegoś powodu wygenerowałby się Wam inny podział, pliki się nadpiszą i git oznaczy różnicę, dlatego warto czasem po puszczeniu uczenia lokalnie zerknąć czy te pliki się nie zmieniły.

Swoje eksperymenty proponuję stawiać na osobnych branczach w osobnych katalogach, forma skryptów uczenia i ewaluacji nie ma znaczenia, ważne, żeby trzymać się podziału zbioru. Po zakończeniu jakiegoś eksperymentu, można merdżować taki katalog do maina + plus krótki komentarz w readme.

<table>
<tr><td>Nazwa</td><td>komentarz</td><td>dice_per_image@nasz_test_set</td><td>ich_dice@ich_val_set (z leader boardu) </td><td>ich_dice@ich_test_set (z leader boardu)</td></tr>
<tr>
    <td>kunet_dk</td>
    <td>xception unet oparty o tutorial z kerasa https://keras.io/examples/vision/oxford_pets_image_segmentation/</td>
    <td>
        1. 0.6894<br />
        2. 0.6764 - wejście rozszerzone o przestrzeń L*a*b, zmiana proporcji bc i jaccard w f. kosztu (na repo)
    </td>
    <td>
        1. 0.5890 <br /> https://dfuc2022.grand-challenge.org/evaluation/95564700-22ee-40ee-bd96-c38455ef1f22/
        2. 0.5969 <br /> https://dfuc2022.grand-challenge.org/evaluation/07e57526-d233-4207-ba19-650afe7ff4a0/ 
    </td>
    <td>-</td>
</tr>
<tr>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</table>
