var myApp = angular.module('myApp',['infinite-scroll','rzModule',"ngSanitize"]);

myApp.controller('DataController', function($scope,$http) {
   $http.get("/s/").then(function(r) {
                    $scope.data = r.data.unknown_emails;
                    $scope.data_main = r.data.unknown_emails;
            });
   $scope.onMouseOver = function(user){
        console.log("mouse over user: " + user.message_id)


        $scope.eFrom = user.header_from;
        $scope.eSent = user.header_date;
        $scope.eTo = user.header_to;
        $scope.eSubject = user.header_subject;
        $scope.eContent = user.email_body;
    }
     $http.get("/controleboard/").then(function(r) {
                    $scope.form = r.data
                    console.log(r.data)

            });

   $scope.onChangeClass0 = function(user){
        user.class1 = !user.class0}
   $scope.onChangeClass1 = function(user){
        user.class0 = !user.class1}

   $scope.onDetails = function(user) {
        console.log("Get details of: " + user.message_id );
        $http({
            method : 'POST',
            url : '/email_data/',
            data : angular.copy({"mail_id":user.message_id} )
        });
    };

   $scope.onSubmit = function(user) {
        user.is_corrected = 'blue'
        console.log("Checked box of user id: " + user.message_id+ " "+user.is_corrected );
        $http({
            method : 'POST',
            url : '/controleboard/',
            data : angular.copy({"mail_id":user.message_id, "truth_class":( +(user.class1==1) ).toString() } )
        });
    };


   $scope.onSelectTab_main = function() {
        console.log("New Tab Selected: ");
        $scope.data = $scope.data_main;
    };
    $scope.onSelectTab_class0 = function() {
        console.log("New Tab Selected: ");
        $scope.data = $scope.data_class0;
    };
   $scope.onSelectTab_class1 = function() {
        console.log("New Tab Selected: ");
        $scope.data = $scope.data_class1;
    };
});

myApp.controller('SliderController', SliderController);

function SliderController($scope,$http) {
  var vm = this;

$scope.performance_images = {mnb:{cm:"/static/Images/confusion_matrices_NA/mnb/cm.png",
            wc: "/static/Images/wordcloud_NA/mnb/wc.png",
            pie: "/static/Images/pies_NA/mnb/pie.png",
            roc: "/static/Images/rocs_NA/mnb/roc.png" },
            rf:{cm:"/static/Images/confusion_matrices_NA/rf/cm.png",
            wc: "/static/Images/wordcloud_NA/rf/wc.png",
            pie: "/static/Images/pies_NA/rf/pie.png",
            roc: "/static/Images/rocs_NA/rf/roc.png" },
            etr:{cm:"/static/Images/confusion_matrices_NA/etr/cm.png",
            wc: "/static/Images/wordcloud_NA/etr/wc.png",
            pie: "/static/Images/pies_NA/etr/pie.png",
            roc: "/static/Images/rocs_NA/etr/roc.png" }}



  $http.get("/images/").then(function(r) {
   $scope.availableImages = r.data;
   $scope.pie = $scope.availableImages.pie;
   console.log($scope.pie)
            });

  $scope.current_model = 'mnb'
  $scope.modelChoices = [
      {name:'Multinomial Naive Bayes', value:2,key:'mnb' },
      {name:'Random Forest', value:1 ,key:'rf'},
      {name:'Extreme Random Forest', value:0,key:'etr' }
    ];


  $scope.trainModel = function() {
    console.log("Retraining the model..." );
        $http({
            method : 'POST',
            url : '/global_performances/',
            data : angular.copy({"message":"AUTO_TRAIN","model_name":$scope.current_model})
        });
   };
  $scope.testButton = function() {
    console.log("Submitting Model changes..." );
            $http({
            method : 'POST',
            url : '/global_performances/',
            data : angular.copy({"message":"TRAIN",
                                 "weight_taak":$scope.val_weight_taak,
                                  "weight_non_taak":$scope.val_weight_non_taak,
                                  "threshold":$scope.val_thres,
                                  "model_name":$scope.current_model
                                 })
        });
    console.log($scope.val_weight_taak)

   };
    $scope.updateDatabase = function() {
    console.log("Updating the database..." );
            $http({
            method : 'POST',
            url : '/update_database/',
            data : angular.copy({"message":"UPDATE_DATABASE"
                                 })
        });
    console.log($scope.val_weight_taak)

   };


  $scope.change_model = function(current_model_choice) {
        $scope.currentImageModel = current_model_choice.value
        $scope.current_model = current_model_choice.key
        console.log(current_model_choice)
        console.log($scope.performance_images[$scope.current_model])
        $http({
            method : 'POST',
            url : '/images/',
            data : angular.copy({"new_model":current_model_choice.key})
        });
         $http({
            method : 'POST',
            url : '/global_performances/',
            data : angular.copy({"new_model":current_model_choice.key})
        });
      };


}

myApp.controller('EmailController', EmailController);
function EmailController($scope,$http,$sce) {
  $scope.modelChoices = [
      {name:'Multinomial Naive Bayes', value:2,key:'mnb' },
      {name:'Random Forest', value:1 ,key:'rf'},
      {name:'Extreme Random Forest', value:0,key:'etr' }
    ];

  $scope.change_model = function(current_model_choice) {
        console.log(current_model_choice)
        $scope.availableEmails = $scope.jsonEmails[current_model_choice.key];

      };

    $scope.reset = function() {
    console.log("Resetting images..." );
        $http({
            method : 'POST',
            url : '/email_data/',
            data : angular.copy({"message":"RESET"})
        });
        }

  $http.get("/email_data/").then(function(r) {
  $scope.jsonEmails = r.data
   $scope.availableEmails = $scope.jsonEmails['mnb'];

            });

  $scope.getColorLabel = function(label){
    if (label == 'TAAK'){
         return 'blue'
    }
    else{
        if (label == 'NONE')
            return 'black'
        return 'red'
    }
  }
  $scope.onMouseOver = function(mail){
    console.log("mouse over mail")

    $scope.pred = mail.pred;
    $scope.truth = mail.truth;
    $scope.emailHTML  = mail.html_body;
    $scope.eFimp_image = mail.eFimp;
    $scope.epie_image = mail.epie;
    if (mail.pred == 'TAAK'){
         $scope.pred_color = 'blue'
    }
    else{
        $scope.pred_color = 'red'
    }
    if (mail.truth == 'TAAK'){
         $scope.truth_color = 'blue'
    }
    else{
     if (mail.truth == 'NONE'){
        $scope.truth_color = 'black'
        }
      else{
        $scope.truth_color = 'red'
        }
    }
}

   $scope.uCanTrust = function(string){
        return $sce.trustAsHtml(string);
    }
 }
