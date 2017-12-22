(function($) {
    // Spammable Model UI
    sp_is_enabled_cb = $("input[name$='spam_model_is_enabled']");
    sp_is_enabled_cb.change(function() {
        ie_cb = $(this);
        ie_next_fields = ie_cb.parent().parent().nextAll()
        data_columns_fieldset = ie_cb.closest("fieldset").next();
        labels_fieldset = data_columns_fieldset.next();

        if (ie_cb.is(':checked')) { 
            ie_next_fields.show()
            data_columns_fieldset.hide();
            labels_fieldset.hide();
        } else { 
            ie_next_fields.hide()
            data_columns_fieldset.show();
            labels_fieldset.show();
        }
    });
    sp_is_enabled_cb.change();
    // BoW UI
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
    
    // CV UI
    cv_is_enabled_cb = $("input[name$='cv_is_enabled']");
    cv_is_enabled_cb.change(function() {
        ie_cb = $(this);
        ie_next_fields = ie_cb.parent().parent().nextAll()

        if (ie_cb.is(':checked')) { 
            ie_next_fields.show()
        } else { 
            ie_next_fields.hide()
        }
    });
    cv_is_enabled_cb.change();

    $(document).on('formset:added', function(event, $row, formsetName) {
        //
    });

    $(document).on('formset:removed', function(event, $row, formsetName) {
        // Row removed, nothing to do here yet
    });
    
})(django.jQuery);
