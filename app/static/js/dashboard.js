document.querySelector('.profile').addEventListener('click', function() {
    document.getElementById('file-input').click(); // Trigger the file input click
});

document.getElementById('file-input').addEventListener('change', function(event) {
    const file = event.target.files[0]; // Get the selected file
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            // You can set the image source to the profile icon if needed
            document.querySelector('.profile').style.backgroundImage = `url(${e.target.result})`;
            document.querySelector('.profile').style.backgroundSize = 'cover'; // Cover the whole area
            document.querySelector('.profile').style.backgroundPosition = 'center'; // Center the image
        };
        reader.readAsDataURL(file); // Read the file as a data URL
    }
});