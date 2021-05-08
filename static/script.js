// const idxx = document.getElementById("idxx");
// idxx.style.color = "green";

// const input_keyword = document.querySelector(".input-keyword");
// const input_keyword = document.querySelector(".sentiment");

// function ubahWarna() {
//   input_keyword.style.backgroundColor = "green";
// }

// input_keyword.onclick = ubahWarna;

// const els = document.querySelector(".display-4");
// els.addEventListener("click", function () {
//   const sectiona = document.querySelector("sectiona");
//   const h3Baru = document.createElement("div");
//   const teksBaru = document.createTextNode("teksbaru");
//   h3Baru.appendChild(teksBaru);
//   sectiona.appendChild(h3Baru);
// });

// input_keyword.keydown = ubahWarna;

// function untuk ubah warna - it works
// const els = document.querySelector(".display-4");
// function ubahWarna() {
//   els.style.backgroundColor = "green";
// }
// els.onclick = ubahWarna;

// PR: cara menambah elemen baru pada halaman html

console.log('trial of vertical scroll table')



$(document).ready(function () {
    // on page load this will fetch data from our flask-app asynchronously
   $.ajax({url: '/word_cloud', success: function (data) {
       // returned data is in string format we have to convert it back into json format
       var words_data = $.parseJSON(data);
       // we will build a word cloud into our div with id=word_cloud
       // we have to specify width and height of the word_cloud chart
       $('#word_cloud').jQCloud(words_data, {
           width: 800,
           height: 600
       });
   }});
});