# 🧵 Le Fil Rouge — mode d'emploi

Une lettre-cadeau à faire défiler : une enveloppe à ouvrir avec un mot de
passe, puis un fil rouge qui traverse toute la page à travers six chapitres
(intro, cadeaux, films, nourriture, activités, conclusion).

Tout est dans 3 fichiers :
- `index.html` → le contenu et le texte
- `style.css` → l'apparence (couleurs, polices, mise en page)
- `script.js` → le mot de passe et les listes de cadeaux/films/activités

Tu n'as besoin d'aucun logiciel spécial : un simple éditeur de texte
(Bloc-notes, TextEdit, ou [VS Code](https://code.visualstudio.com/) en mieux)
suffit pour tout personnaliser.

---

## 1. Voir le résultat tout de suite

Double-clique sur `index.html` : il s'ouvre dans ton navigateur. C'est tout.
Tu peux le modifier, enregistrer, puis rafraîchir la page (F5) pour voir les
changements.

---

## 2. Changer le mot de passe

Ouvre `script.js`, tout en haut :

```js
const PASSWORD = "changemoi";
```

Remplace `"changemoi"` par votre mot secret (garde les guillemets). La
comparaison ignore les majuscules/minuscules et les espaces autour, donc
pas besoin d'être exact à la lettre près.

Tu peux aussi remplir `PASSWORD_HINT` avec un indice qui s'affichera après
3 essais ratés, par exemple `"notre premier resto"`.

⚠️ **Ce n'est pas un vrai coffre-fort** : n'importe qui sachant regarder le
code source du fichier peut retrouver le mot de passe. C'est un geste
romantique, pas une protection bancaire — largement suffisant pour ce cadeau,
mais ne mets rien de sensible derrière.

---

## 3. Personnaliser les textes

Ouvre `index.html` et cherche les commentaires `<!-- ✏️ PERSONNALISE-MOI -->` :
- **Chapitre I** (introduction) : le texte d'ouverture de la lettre
- **Chapitre VI** (conclusion) : le texte de fin + `[Ton prénom]` à remplacer
  dans la ligne de signature

Le reste du texte (titres de chapitres, etc.) peut aussi être changé
directement dans le HTML, il n'y a pas de piège : c'est juste du texte entre
des balises.

---

## 4. Les cadeaux (Chapitre II)

Dans `script.js` :

```js
const gifts = [
  { emoji: "🎁", title: "Cadeau n°1", text: "..." },
  { emoji: "🎀", title: "Cadeau n°2", text: "..." },
];
```

- Modifie `title` et `text` pour chaque cadeau
- Ajoute ou supprime des lignes pour changer le nombre de cadeaux
- `emoji` s'affiche sur la face fermée du paquet, avant qu'on clique dessus

---

## 5. Les films (Chapitre III)

1. Récupère les affiches des films dont vous avez parlé (capture d'écran,
   téléchargement depuis un site licite, etc.) en `.jpg` ou `.png`
2. Dépose-les dans le dossier `images/movies/`
3. Dans `script.js`, adapte la liste :

```js
const movies = [
  { title: "Titre du film", image: "images/movies/film-1.jpg" },
];
```

Le nom dans `image` doit correspondre exactement au nom du fichier déposé.
Si une affiche manque ou que le nom ne correspond pas, une carte de
remplacement avec juste le titre s'affiche automatiquement — rien ne
« casse » visuellement.

---

## 6. La nourriture (Chapitre IV)

Directement dans `index.html`, cherche `id="food"`. Chaque plat est une
ligne comme :

```html
<li><span class="food-emoji">🍣</span>Sushi</li>
```

Copie/colle une ligne pour en ajouter, ou supprime-en une. Il y a deux
colonnes : « On dit toujours oui » et « On évite soigneusement ».

---

## 7. Les activités (Chapitre V)

Dans `script.js` :

```js
const activities = [
  { emoji: "👩‍🍳", label: "Cuisiner ensemble" },
];
```

Ajoute, retire ou renomme les activités. Sur la page, cliquer sur une carte
la met en surbrillance (pour dire « j'ai envie de celle-ci ») — c'est juste
un effet visuel pendant la lecture, rien n'est enregistré.

---

## 8. Partager la lettre

Trois façons de faire, du plus simple au plus « propre » :

**A. Envoyer le dossier tel quel**
Zippe le dossier `cadeau-anniversaire` et envoie-le. La personne double-clique
sur `index.html` après extraction. Fonctionne partout, aucune installation.

**B. Héberger en ligne (accessible par un lien, sur mobile aussi)**
- [Netlify Drop](https://app.netlify.com/drop) : glisse-dépose le dossier
  entier sur la page, tu obtiens un lien en quelques secondes, gratuit, sans
  compte obligatoire.
- [GitHub Pages](https://pages.github.com/) : un peu plus technique mais
  gratuit et durable si tu es à l'aise avec Git.

**C. L'ouvrir en direct sur ton propre téléphone/ordinateur** le jour J,
sans rien héberger — l'option A suffit largement pour ça.

---

## 9. Détails techniques (facultatif)

- Aucune dépendance à installer, aucun serveur nécessaire : HTML/CSS/JS pur.
- Les polices (Fraunces, Newsreader, Homemade Apple) viennent de Google
  Fonts et nécessitent une connexion internet pour s'afficher parfaitement ;
  sans connexion, le navigateur utilise une police de secours proche.
- Le site respecte les préférences de réduction des animations
  (`prefers-reduced-motion`) et reste utilisable au clavier.
- Responsive : testé mentalement du format mobile étroit jusqu'au grand
  écran ; le fil rouge devient une barre horizontale en haut de l'écran sur
  mobile.

Joyeux anniversaire à elle/lui — et bon fil rouge. 🧵❤️
