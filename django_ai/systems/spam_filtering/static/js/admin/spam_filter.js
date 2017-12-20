(function($) {
    bow_is_enabled_cb = $("input[name$='bow_is_enabled']");
    bow_is_enabled_cb.change(function() {
        ie_cb = $(this);
        ie_next_fields = ie_cb.parent().parent().nextAll()
        bow_misc_fieldset = ie_cb.closest("fieldset").next();

        if (ie_cb.is(':checked')) { 
            ie_next_fields.show()
            bow_misc_fieldset.show();
        } else { 
            ie_next_fields.hide()
            bow_misc_fieldset.hide();
        }
    });
    bow_is_enabled_cb.change();

    $(document).on('formset:added', function(event, $row, formsetName) {
        //
    });

    $(document).on('formset:removed', function(event, $row, formsetName) {
        // Row removed, nothing to do here yet
    });
    
})(django.jQuery);
