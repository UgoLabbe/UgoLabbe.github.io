/* ============================================================================
   🔐 PERSONNALISATION — mot de passe de la lettre
   ============================================================================ */
const PASSWORD = "11/07/2026";      // <-- remplace par votre mot secret (pas sensible à la casse/espaces)
const PASSWORD_HINT = "";          // optionnel : indice affiché après 3 essais ratés, ex. "notre chanson"

/* ============================================================================
   🎁 PERSONNALISATION — Chapitre II, les cadeaux
   Ajoute / supprime des lignes selon le nombre de cadeaux que tu as.
   ============================================================================ */
const gifts = [
  { emoji: "🫂", title: "Beaucoup de calins", text: "Pour se sentir toujours plus proches, toujours aimés, et faire disparaître les doutes et les problèmes." },
  { emoji: "😚", title: "Enormement de bisous", text: "Pour renforcer notre lien et exprimer notre amour." },
  { emoji: "💭", title: "Pleins de rêves", text: "Pour imaginer notre futur ensemble, espérer pour l'avenir et le construire ensemble." },
];

/* ============================================================================
   🎬 PERSONNALISATION — Chapitre III, les films
   Dépose tes affiches (.jpg/.png) dans le dossier images/movies/
   puis indique ici leur nom de fichier exact. Si une image manque,
   une carte de secours s'affiche automatiquement avec le titre.
   ============================================================================ */
const movies = [
  { title: "Om Shanti Om", image: "images/movies/om-shanti-om.jpg" },
  { title: "Good will hunting", image: "images/movies/good-will.jpg" },
  { title: "Pride and Prejudice", image: "images/movies/pride-and-prej.jpg" },
  { title: "Wall-E", image: "images/movies/wall-e.jpg"},
  { title: "Raiponce", image: "images/movies/raiponce.jpg"},
  { title: "The office", image: "images/movies/the-office.jpg"},
  { title: "The Notebook", image: "images/movies/the-notebook.jpg"},
  { title: "Barbie", image: "images/movies/barbie.jpg"},
  { title: "Notting Hill", image: "images/movies/notting-hill.jpg"},
  { title: "Les Harry Potter!", image: "images/movies/harry-potter.jpg"},
  { title: "Project Hail Mary", image: "images/movies/hail-mary.jpg"},
  { title: "Les Miyazaki!", image: "images/movies/miyazaki.jpg"},
  { title: "The Grand Budapest Hotel", image: "images/movies/the-grand-budapest-hotel.jpg"},
];

/* ============================================================================
   💫 PERSONNALISATION — Chapitre V, les activités / dates
   ============================================================================ */
const activities = [
  { emoji: "👩‍🍳", label: "Cuisiner ensemble" },
  { emoji: "🎤", label: "Soirée karaoké" },
  { emoji: "🖼️", label: "Faire une sortie musée" },
  { emoji: "🪂", label: "Sauter à l'élastique (contre mon gré)" },
  { emoji: "🌅", label: "Regarder un beau coucher de soleil" },
  { emoji: "📖", label: "Lire un beau livre de poésie ensemble, et se reconnaître dans les textes" },
  { emoji: "✨", label: "À inventer ensemble" },
];

if (PASSWORD === "changemoi") {
  console.warn("⚠️ N'oublie pas de changer le mot de passe dans script.js avant d'envoyer la lettre !");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/* ---------------------------------------------------------------------------
   Rendu du contenu
   --------------------------------------------------------------------------- */
function renderGifts() {
  const grid = document.getElementById("giftsGrid");
  if (!grid) return;
  grid.innerHTML = gifts.map((g) => `
    <div class="gift-card">
      <div class="gift-inner">
        <div class="gift-face gift-front">
          <span class="gift-emoji">${g.emoji}</span>
          <span class="gift-cta">touche-moi</span>
        </div>
        <div class="gift-face gift-back">
          <p class="gift-title">${escapeHtml(g.title)}</p>
          <p class="gift-text">${escapeHtml(g.text)}</p>
        </div>
      </div>
    </div>
  `).join("");

  grid.querySelectorAll(".gift-card").forEach((card) => {
    card.addEventListener("click", () => card.classList.toggle("open"));
  });
}

function renderMovies() {
  const grid = document.getElementById("moviesGrid");
  if (!grid) return;
  grid.innerHTML = movies.map((m) => `
    <div class="movie-card">
      <img src="${m.image}" alt="Affiche de ${escapeHtml(m.title)}" loading="lazy">
      <div class="movie-title-tag">${escapeHtml(m.title)}</div>
    </div>
  `).join("");

  grid.querySelectorAll(".movie-card").forEach((card, i) => {
    const img = card.querySelector("img");
    img.addEventListener("error", () => {
      img.remove();
      const fallback = document.createElement("div");
      fallback.className = "movie-fallback";
      fallback.innerHTML = `<span class="clap">🎬</span><span>${escapeHtml(movies[i].title)}</span>`;
      card.prepend(fallback);
    }, { once: true });
  });
}

function renderActivities() {
  const grid = document.getElementById("activitiesGrid");
  if (!grid) return;
  grid.innerHTML = activities.map((a) => `
    <div class="activity-card">
      <span class="activity-emoji">${a.emoji}</span>
      <span class="activity-label">${escapeHtml(a.label)}</span>
      <span class="activity-check">✓ envie</span>
    </div>
  `).join("");

  grid.querySelectorAll(".activity-card").forEach((card) => {
    card.addEventListener("click", () => card.classList.toggle("picked"));
  });
}

/* ---------------------------------------------------------------------------
   Accordéon des films
   --------------------------------------------------------------------------- */
function initMoviesAccordion() {
  const toggle = document.getElementById("moviesToggle");
  const panel = document.getElementById("moviesPanel");
  if (!toggle || !panel) return;
  toggle.addEventListener("click", () => {
    const open = panel.classList.toggle("open");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    toggle.querySelector("span").textContent = open ? "Cacher la liste" : "Voir la liste";
  });
}

/* ---------------------------------------------------------------------------
   Portail : enveloppe + mot de passe
   --------------------------------------------------------------------------- */
function initGate() {
  const envelope = document.getElementById("envelope");
  const gateStage = document.getElementById("gateStage");
  const passwordStage = document.getElementById("passwordStage");
  const passwordInput = document.getElementById("passwordInput");
  const passwordRow = document.querySelector(".password-row");
  const passwordError = document.getElementById("passwordError");
  const passwordHint = document.getElementById("passwordHint");
  const passwordSubmit = document.getElementById("passwordSubmit");
  const toggleBtn = document.getElementById("togglePassword");

  let attempts = 0;

  envelope.addEventListener("click", () => {
    if (envelope.classList.contains("opening")) return;
    envelope.classList.add("opening");
    setTimeout(() => {
      gateStage.hidden = true;
      passwordStage.hidden = false;
      passwordStage.classList.add("fade-in-up");
      passwordInput.focus();
    }, 550);
  });

  toggleBtn.addEventListener("click", () => {
    const isHidden = passwordInput.type === "password";
    passwordInput.type = isHidden ? "text" : "password";
    toggleBtn.textContent = isHidden ? "🙈" : "👁";
    passwordInput.focus();
  });

  function tryUnlock() {
    const value = passwordInput.value.trim().toLowerCase();
    if (value.length && value === PASSWORD.trim().toLowerCase()) {
      unlockLetter();
      return;
    }
    attempts++;
    passwordRow.classList.remove("shake");
    void passwordRow.offsetWidth; // relance l'animation
    passwordRow.classList.add("shake");
    passwordError.classList.add("show");
    passwordInput.value = "";
    passwordInput.focus();
    if (attempts >= 3 && PASSWORD_HINT) {
      passwordHint.textContent = "Indice : " + PASSWORD_HINT;
      passwordHint.classList.add("show");
    }
  }

  passwordSubmit.addEventListener("click", tryUnlock);
  passwordInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") tryUnlock();
  });

  function unlockLetter() {
    document.body.classList.remove("locked");
    document.body.classList.add("unlocked");
    const gate = document.getElementById("gate");
    setTimeout(() => { gate.style.display = "none"; }, 800);
    initScrollEffects();
  }
}

/* ---------------------------------------------------------------------------
   Fil rouge de progression + apparition des chapitres au scroll
   --------------------------------------------------------------------------- */
let scrollEffectsInitialized = false;

function updateThreadProgress() {
  const fill = document.getElementById("threadFill");
  if (!fill) return;
  const scrollTop = window.scrollY || document.documentElement.scrollTop;
  const docHeight = document.documentElement.scrollHeight - window.innerHeight;
  const progress = docHeight > 0 ? Math.min(1, Math.max(0, scrollTop / docHeight)) : 0;
  fill.style.setProperty("--progress", progress);
}

function initScrollEffects() {
  if (scrollEffectsInitialized) return;
  scrollEffectsInitialized = true;

  updateThreadProgress();
  window.addEventListener("scroll", () => {
    requestAnimationFrame(updateThreadProgress);
  }, { passive: true });

  const chapters = document.querySelectorAll(".chapter");
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.15 });
    chapters.forEach((ch) => observer.observe(ch));
  } else {
    chapters.forEach((ch) => ch.classList.add("in-view"));
  }
}

/* ---------------------------------------------------------------------------
   Initialisation
   --------------------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  renderGifts();
  renderMovies();
  renderActivities();
  initMoviesAccordion();
  initGate();
});
