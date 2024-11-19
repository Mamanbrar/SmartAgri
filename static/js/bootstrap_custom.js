// JavaScript for Carousel
document.addEventListener("DOMContentLoaded", function() {
    const carousels = document.querySelectorAll(".carousel");
    carousels.forEach(carousel => {
        const items = carousel.querySelectorAll(".carousel-item");
        let currentIndex = 0;
        
        const next = carousel.querySelector(".carousel-control-next");
        const prev = carousel.querySelector(".carousel-control-prev");
        
        function showItem(index) {
            items.forEach((item, i) => item.classList.toggle("active", i === index));
        }
        
        next.addEventListener("click", () => {
            currentIndex = (currentIndex + 1) % items.length;
            showItem(currentIndex);
        });
        
        prev.addEventListener("click", () => {
            currentIndex = (currentIndex - 1 + items.length) % items.length;
            showItem(currentIndex);
        });
    });
});

// JavaScript for Dropdown
document.querySelectorAll('.dropdown').forEach(dropdown => {
    dropdown.addEventListener('mouseenter', function() {
        this.querySelector('.dropdown-content').style.display = 'block';
    });
    dropdown.addEventListener('mouseleave', function() {
        this.querySelector('.dropdown-content').style.display = 'none';
    });
});
